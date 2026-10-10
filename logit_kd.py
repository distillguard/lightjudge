"""Offline full-vocabulary answer-position Logit KD; shared hard EOS formatting loss."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import pilot
from teacher_quantization import ROOT, CACHE, MODEL_ID, REVISION, load_text_model

OUTPUT = ROOT / "outputs" / "logit_kd"
TEMPERATURE = 2.0
VOCAB_SIZE = 248320
LOSS = "0.5 * (T^2 forward KL at answer position + shared hard EOS cross-entropy)"


def distillation_loss(logits, targets, teacher_log_probs, variant="forward", metrics=None):
    import torch
    import torch.nn.functional as F
    assert logits.ndim == 2 and logits.shape[0] == targets.numel() == 2
    teacher_log_probs = teacher_log_probs.to(device=logits.device, dtype=torch.float32)
    assert teacher_log_probs.shape == logits[0].shape and not teacher_log_probs.requires_grad
    log_q = F.log_softmax(logits[0].float()/TEMPERATURE, dim=-1)
    forward = F.kl_div(log_q, teacher_log_probs, reduction="sum", log_target=True)
    if variant == "forward" and metrics is None:
        soft = forward
    else:
        reverse = (log_q.exp() * (log_q - teacher_log_probs)).sum()
        c_p, c_q, gate = confidence_gate(teacher_log_probs, log_q)
        if variant == "forward":
            weight = torch.ones_like(gate)
        elif variant == "reverse_kl":
            weight = torch.zeros_like(gate)
        elif variant == "fixed_mix":
            weight = torch.full_like(gate, 0.5)
        elif variant == "adaptive_mix":
            weight = gate
        else:
            raise ValueError("Unknown KL variant")
        soft = weight * forward + (1-weight) * reverse
        if metrics is not None:
            values = torch.stack([forward, reverse, c_p, c_q, gate, weight]).detach().cpu().tolist()
            metrics.update(dict(zip(("forward_kl", "reverse_kl", "teacher_confidence",
                                     "student_confidence", "adaptive_gate", "forward_weight"), values)))
    eos = F.cross_entropy(logits[1:].float(), targets[1:])
    if metrics is not None:
        metrics["eos_ce"] = eos.detach().item()
    return 0.5 * (soft * TEMPERATURE**2 + eos)


def confidence_gate(log_p, log_q):
    import math
    import torch
    assert log_p.shape == log_q.shape and log_p.ndim == 1 and log_p.numel() > 1
    denominator = math.log(log_p.numel())
    c_p = 1 + (log_p.exp() * log_p).sum()/denominator
    c_q = 1 + (log_q.exp() * log_q).sum()/denominator
    gate = (c_p >= c_q).to(dtype=torch.float32).detach()
    return c_p, c_q, gate


def prefix_hash(example):
    prefix = example["input_ids"][:example["prompt_tokens"]]
    return hashlib.sha256(json.dumps(prefix).encode()).hexdigest()


def extract():
    import torch
    import train_student
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, BitsAndBytesConfig
    path, summary_path = OUTPUT/"teacher_log_probs.safetensors", OUTPUT/"soft_targets_summary.json"
    if path.exists() or summary_path.exists():
        raise FileExistsError("Existing soft targets preserved")
    rows, teacher = train_student.load_training_rows()
    tokenizer = AutoTokenizer.from_pretrained(pilot.STUDENT_ID, revision=pilot.STUDENT_REVISION,
        cache_dir=str(CACHE),local_files_only=True,trust_remote_code=False)
    teacher_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
        cache_dir=str(CACHE),local_files_only=True,trust_remote_code=False)
    assert tokenizer.get_vocab() == teacher_tokenizer.get_vocab()
    assert tokenizer.special_tokens_map == teacher_tokenizer.special_tokens_map
    examples = [train_student.encode_example(row,"response_kd",tokenizer) for row in rows]
    metadata = {"status":"running","source_split":"train","sample_ids":[r["source_id"] for r in rows],
        "rows":len(rows),"train_sha256":teacher["data_sha256"],"teacher_responses_sha256":teacher["output_sha256"],
        "prefix_sha256":[prefix_hash(e) for e in examples],"model_id":MODEL_ID,"revision":REVISION,
        "quantization":"NF4 double quant, BF16 compute","temperature":TEMPERATURE,"vocabulary_size":VOCAB_SIZE,
        "stored_dtype":"FP32","scope":"full vocabulary at first answer position","gold_used_in_input":False,
        "gold_used_to_filter":False,"test_evaluation":False,"started_at_kst":train_student.now(),
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    pilot.write_json(summary_path,metadata)
    try:
        quant = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        start=perf_counter()
        model,_=load_text_model(MODEL_ID,REVISION,quant)
        model.eval().requires_grad_(False)
        model.config.use_cache=False
        metadata["load_seconds"]=round(perf_counter()-start,3)
        torch.cuda.reset_peak_memory_stats()
        start=perf_counter()
        stored=torch.empty((len(rows),VOCAB_SIZE),dtype=torch.float32,device="cpu")
        reviews=[]
        for i,(row,example) in enumerate(zip(rows,examples)):
            prefix=example["input_ids"][:example["prompt_tokens"]]
            ids=torch.tensor([prefix],device="cuda",dtype=torch.long)
            with torch.inference_mode():
                logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),
                             logits_to_keep=1,use_cache=False).logits[0,0].float()
                assert logits.numel()==VOCAB_SIZE and torch.isfinite(logits).all()
                log_probs=(logits/TEMPERATURE).log_softmax(-1)
                stored[i].copy_(log_probs.cpu())
                probabilities=logits.softmax(-1)
                choice_ids=[tokenizer.encode(c["label"],add_special_tokens=False)[0] for c in row["choices"]]
                reviews.append({"source_id":row["source_id"],"teacher_target":row["teacher_target"],
                    "gold":row["gold"],"full_vocab_top1_id":int(logits.argmax()),
                    "hard_answer_id":example["target_ids"][0],
                    "choice_probabilities_T1":{c["label"]:float(probabilities[j]) for c,j in zip(row["choices"],choice_ids)},
                    "choice_probabilities_T2":{c["label"]:float(log_probs[j].exp()) for c,j in zip(row["choices"],choice_ids)},
                    "choice_mass_T1":float(probabilities[choice_ids].sum()),
                    "entropy_T2":float(-(log_probs.exp()*log_probs).sum())})
            if (i+1)%32==0 or i+1==len(rows):
                metadata["completed"]=i+1
                pilot.write_json(summary_path,metadata)
                print(f"Teacher soft targets: {i+1}/{len(rows)}",flush=True)
        assert torch.isfinite(stored).all() and torch.allclose(stored.logsumexp(-1),torch.zeros(len(rows)),atol=1e-5)
        save_file({"answer_log_probs":stored.contiguous()},str(path))
        pilot.write_rows(OUTPUT/"soft_target_review.jsonl",reviews)
        metadata.update({"status":"passed","extraction_seconds":round(perf_counter()-start,3),
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"output_bytes":path.stat().st_size,
            "output_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "top1_matches_saved_response":sum(x["full_vocab_top1_id"]==x["hard_answer_id"] for x in reviews),
            "min_choice_mass_T1":min(x["choice_mass_T1"] for x in reviews),
            "mean_entropy_T2":sum(x["entropy_T2"] for x in reviews)/len(reviews)})
    except Exception as exc:
        metadata.update({"status":"failed","error":f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        metadata["finished_at_kst"]=train_student.now()
        pilot.write_json(summary_path,metadata)


def load_targets(examples):
    import torch
    from safetensors.torch import load_file
    summary=json.loads((OUTPUT/"soft_targets_summary.json").read_text(encoding="utf-8"))
    assert summary["status"]=="passed" and summary["source_split"]=="train"
    assert summary["model_id"]==MODEL_ID and summary["revision"]==REVISION
    assert summary["temperature"]==TEMPERATURE and summary["vocabulary_size"]==VOCAB_SIZE
    assert summary["sample_ids"]==[e["source_id"] for e in examples]
    assert summary["prefix_sha256"]==[prefix_hash(e) for e in examples]
    path=OUTPUT/"teacher_log_probs.safetensors"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==summary["output_sha256"]
    values=load_file(str(path),device="cpu")["answer_log_probs"]
    assert values.shape==(len(examples),VOCAB_SIZE) and values.dtype==torch.float32
    assert torch.isfinite(values).all() and torch.allclose(values.logsumexp(-1),torch.zeros(len(examples)),atol=1e-5)
    return values,summary


def probe():
    import torch
    import train_student
    from transformers import set_seed
    path=OUTPUT/"memory_probe.json"
    if path.exists():
        raise FileExistsError("Existing Logit probe preserved")
    config,config_hash=train_student.run_config("logit_kd",42)
    report={"status":"running","config":config,"config_sha256":config_hash,
            "probe_model_reused_for_training":False,"started_at_kst":train_student.now()}
    pilot.write_json(path,report)
    try:
        model,tokenizer,parameters,optimizer=train_student.load_student(42)
        rows,_=train_student.load_training_rows()
        examples=[train_student.encode_example(r,"logit_kd",tokenizer) for r in rows]
        values,summary=load_targets(examples)
        index=max(range(len(examples)),key=lambda i:len(examples[i]["input_ids"]))
        initial=train_student.adapter_hash(model)
        set_seed(42)
        torch.cuda.reset_peak_memory_stats()
        start=perf_counter()
        for _ in range(config["gradient_accumulation"]):
            loss=train_student.answer_loss(model,examples[index],values[index])
            (loss/config["gradient_accumulation"]).backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,config["max_grad_norm"],error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        assert train_student.adapter_hash(model)!=initial and model.get_input_embeddings().weight.grad is None
        report.update({"status":"passed","source_id":examples[index]["source_id"],
            "sequence_tokens":len(examples[index]["input_ids"]),"loss":loss.item(),"grad_norm":float(norm),
            "seconds":round(perf_counter()-start,3),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "soft_targets_sha256":summary["output_sha256"],"trainable_parameters":sum(p.numel() for p in parameters)})
    except Exception as exc:
        report.update({"status":"failed","error":f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        report["finished_at_kst"]=train_student.now()
        pilot.write_json(path,report)
        print(json.dumps(report,ensure_ascii=False),flush=True)


def self_check():
    import torch
    import torch.nn.functional as F
    teacher=torch.tensor([1.,-2.,3.],requires_grad=False)
    student=torch.tensor([[2.,0.,-1.],[1.,3.,-2.]],requires_grad=True)
    targets=torch.tensor([2,1])
    log_p=(teacher/TEMPERATURE).log_softmax(-1)
    loss=distillation_loss(student,targets,log_p)
    log_q=(student[0]/TEMPERATURE).log_softmax(-1)
    direct=0.5*((log_p.exp()*(log_p-log_q)).sum()*TEMPERATURE**2+F.cross_entropy(student[1:],targets[1:]))
    assert torch.allclose(loss,direct)
    actual=torch.autograd.grad(loss,student,retain_graph=True)[0]
    expected=torch.autograd.grad(direct,student)[0]
    assert torch.allclose(actual,expected) and teacher.grad is None
    same=torch.stack([teacher,student[1].detach()])
    zero_soft=distillation_loss(same,targets,log_p)-0.5*F.cross_entropy(same[1:],targets[1:])
    assert abs(float(zero_soft))<1e-6
    changed_targets=targets.clone()
    changed_targets[0]=0
    assert torch.allclose(distillation_loss(student,changed_targets,log_p),loss)
    print("SELF-CHECK PASS: forward KL direction, T^2, position mean, gradients, fixed EOS, no hard/gold answer in loss")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["extract","probe","self-check"])
    args=parser.parse_args()
    {"extract":extract,"probe":probe,"self-check":self_check}[args.action]()
