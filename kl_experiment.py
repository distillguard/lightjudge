"""Fixed ARC KL experiment: confidence, cached targets, probes, training and sealed test."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
import pilot
import train_student as trainer
import logit_kd
import monitor_training as diagnostic
from teacher_quantization import ROOT, CACHE, MODEL_ID, REVISION, load_text_model

OUT = ROOT/"outputs/kl_extension"
CONDITIONS = ("gold_sft", "response_kd", "logit_kd", "reverse_kl", "fixed_mix", "adaptive_mix")
NEW = CONDITIONS[3:]
SEEDS = (42, 7, 123)
SOFT = CONDITIONS[2:]
MONITOR_N = 64
sha = diagnostic.sha


def ready(path):
    path = Path(path)
    if not path.exists():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "passed":
        raise RuntimeError(f"Incomplete result requires inspection: {path}")
    return True


def rows_for(scope, split="train"):
    assert scope in ("pilot", "full") and split in ("train", "validation", "test")
    name = "train_full" if scope == "full" and split == "train" else split
    path = pilot.DATA/f"{name}.jsonl"
    manifest = json.loads((pilot.OUTPUT/"data_manifest.json").read_text(encoding="utf-8"))
    assert sha(path) == manifest["export_sha256"][path.name]
    rows = pilot.read_rows(path)
    assert len(rows) == (1118 if scope == "full" else 256) if split == "train" else len(rows) == (299 if split == "validation" else 1172)
    assert all(r["source_split"] == split for r in rows)
    assert len({r["source_id"] for r in rows}) == len(rows)
    for row in rows:
        assert row["gold"] in {c["label"] for c in row["choices"]}
    return rows, path


def monitor_rows(scope, split):
    rows,_ = rows_for(scope,split)
    return sorted(rows,key=lambda r:hashlib.sha256(r["source_id"].encode()).hexdigest())[:MONITOR_N]


def run_path(scope, condition, seed):
    assert condition in CONDITIONS and seed in SEEDS
    if scope == "pilot" and condition not in NEW:
        return pilot.training_run_path(condition, seed)
    assert scope in ("pilot", "full")
    return OUT/"training"/scope/f"seed_{seed}"/condition


def code_hashes():
    return {name:sha(ROOT/name) for name in
            ("kl_experiment.py", "analyze_kl.py", "train_student.py", "logit_kd.py", "monitor_training.py", "pilot.py", "teacher_quantization.py")}


def require_protocol():
    path = OUT/"training_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["status"] == "passed" and protocol["code_sha256"] == code_hashes()
    return protocol, sha(path)


def check_test(scope, role, condition, seed):
    protocol,_ = require_protocol()
    if scope not in protocol["scopes"]:
        raise ValueError("Test evaluation is restricted to the frozen main-experiment scope")
    path = OUT/"test_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "passed" and manifest["code_sha256"] == code_hashes()
    assert sha(OUT/"training_protocol.json") == manifest["training_protocol_sha256"]
    assert sha(pilot.DATA/"test.jsonl") == manifest["test_data_sha256"]
    tag = f"{scope}/{condition}/{seed}" if condition else f"{role}_baseline"
    assert tag in manifest["models"]
    for filename,checksum in manifest["models"][tag].items():
        assert sha(Path(filename)) == checksum
    return sha(path)


def load_eval_model(role, adapter=None):
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    model_id, revision = (MODEL_ID, REVISION) if role == "teacher" else (pilot.STUDENT_ID, pilot.STUDENT_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=str(CACHE),
                                              local_files_only=True, trust_remote_code=False)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16) if role == "teacher" else None
    started = perf_counter()
    model,_ = load_text_model(model_id, revision, quant)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    snapshot = CACHE/f"models--{model_id.replace('/', '--')}"/"snapshots"/revision
    info = {"load_seconds":perf_counter()-started, "model_id":model_id, "revision":revision,
        "precision":"NF4 double quant, BF16 compute" if role == "teacher" else "BF16 base, FP32 LoRA",
        "loaded_model_footprint_bytes":model.get_memory_footprint(),
        "public_checkpoint_safetensors_bytes":sum(p.stat().st_size for p in snapshot.glob("*.safetensors")),
        "adapter_bytes":0 if adapter is None else sum(p.stat().st_size for p in Path(adapter).glob("*") if p.is_file()),
        "adapter_merged":False, "public_checkpoint_size_is_not_quantized_disk_size":role == "teacher"}
    return model, tokenizer, info


def score(scope, role="student", condition=None, seed=42, split="validation"):
    import torch
    assert role in ("teacher", "student") and (condition is None or role == "student")
    rows,path = rows_for(scope, split)
    test_hash = check_test(scope, role, condition, seed) if split == "test" else None
    tag = f"{scope}_{condition}_{seed}" if condition else f"{role}_baseline"
    folder = OUT/"scores"/split
    summary_path = folder/f"{tag}_summary.json"
    output = folder/f"{tag}.jsonl"
    if ready(summary_path):
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        assert sha(output) == saved["output_sha256"] and saved["data_sha256"] == sha(path)
        if split == "test":
            assert saved["test_manifest_sha256"] == test_hash
        return
    assert not output.exists()
    adapter = None
    if condition:
        run = run_path(scope, condition, seed)
        training = json.loads((run/"summary.json").read_text(encoding="utf-8"))
        assert training["status"] == "passed" and training["config"]["seed"] == seed
        adapter = run/"adapter"
        for name,checksum in training["adapter_files_sha256"].items():
            assert sha(adapter/name) == checksum
    metadata = {"status":"running", "scope":scope, "role":role, "condition":condition, "seed":seed,
        "split":split, "n":len(rows), "data_sha256":sha(path), "code_sha256":code_hashes(),
        "test_manifest_sha256":test_hash, "test_evaluation":split == "test",
        "batch_size":1, "warmup_questions":3, "generation":"thinking=False, greedy, max_new_tokens=8, use_cache=False, output_scores=True",
        "started_at_kst":trainer.now()}
    pilot.write_json(summary_path, metadata)
    try:
        model, tokenizer, loading = load_eval_model(role, adapter)
        metadata.update(loading)
        for row in rows[:3]:
            example = trainer.encode_example(row, "gold_sft", tokenizer)
            ids = torch.tensor([example["input_ids"][:example["prompt_tokens"]]],device="cuda")
            with torch.inference_mode():
                model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=8,
                               do_sample=False,use_cache=False,pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with diagnostic.evaluation_scope(model):
            records, metrics = diagnostic.evaluate(model, tokenizer, rows, split, 3 if condition else 0)
        expected = None
        if split == "validation":
            if condition in CONDITIONS[:3] and scope == "pilot":
                expected = pilot.OUTPUT/f"{pilot.validation_stem(condition,seed)}.jsonl"
            elif condition is None:
                expected = pilot.OUTPUT/f"{role}_baseline.jsonl"
        if expected is not None:
            previous = pilot.read_rows(expected)
            assert [r["source_id"] for r in previous] == [r["source_id"] for r in records]
            assert all(a["predicted"] == b["predicted"] for a,b in zip(previous,records)), "New scores changed original generated answers"
            metadata["original_predictions_match"] = True
        times = [r["generation_seconds"] for r in records]
        metrics["inference"] = {"generation_seconds":sum(times),"latency_mean_seconds":float(np.mean(times)),
            "latency_p50_seconds":float(np.median(times)),"latency_p95_seconds":float(np.quantile(times,.95)),
            "questions_per_second":len(records)/sum(times),
            "generated_tokens_per_second":sum(r["generated_tokens"] for r in records)/sum(times),
            "includes_score_output_overhead":True}
        pilot.write_rows(output, records)
        metadata.update({"status":"passed","metrics":metrics,"output_sha256":sha(output),
                         "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                         "peak_reserved_bytes":torch.cuda.max_memory_reserved()})
    except Exception as exc:
        metadata.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata["finished_at_kst"] = trainer.now()
        pilot.write_json(summary_path, metadata)
    print(json.dumps({"scored":tag,"split":split,"accuracy":metrics["accuracy"],
                     "calibration":metrics["calibration"]},ensure_ascii=False),flush=True)


def prepare_full():
    import torch
    from safetensors.torch import save_file
    folder = OUT/"targets/full"
    summary_path, target_path = folder/"summary.json", folder/"teacher_log_probs.safetensors"
    if ready(summary_path):
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        assert sha(target_path) == saved["output_sha256"] and sha(folder/"teacher_responses.jsonl") == saved["response_sha256"]
        return
    assert not target_path.exists()
    rows,path = rows_for("full")
    metadata = {"status":"running","scope":"full","source_split":"train","n":len(rows),
        "data_sha256":sha(path),"sample_ids":[r["source_id"] for r in rows],"temperature":2.0,
        "vocabulary_size":248320,"gold_used_to_filter":False,"test_evaluation":False,"code_sha256":code_hashes()}
    pilot.write_json(summary_path, metadata)
    try:
        model,tokenizer,loading = load_eval_model("teacher")
        from transformers import AutoTokenizer
        student_tok = AutoTokenizer.from_pretrained(pilot.STUDENT_ID,revision=pilot.STUDENT_REVISION,
                        cache_dir=str(CACHE),local_files_only=True,trust_remote_code=False)
        assert student_tok.get_vocab() == tokenizer.get_vocab()
        assert student_tok.special_tokens_map == tokenizer.special_tokens_map
        probabilities = torch.empty((len(rows),248320),dtype=torch.float32,device="cpu")
        responses,prefix_hashes = [],[]
        started = perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for i,row in enumerate(rows):
            rendered = tokenizer.apply_chat_template([{"role":"user","content":pilot.prompt(row)}],
                              tokenize=False,add_generation_prompt=True,enable_thinking=False)
            prefix = tokenizer.encode(rendered,add_special_tokens=False)
            assert len(prefix)+2 <= trainer.CONFIG["max_sequence_length"]
            prefix_hashes.append(hashlib.sha256(json.dumps(prefix).encode()).hexdigest())
            ids = torch.tensor([prefix],device="cuda")
            with torch.inference_mode():
                logits = model(input_ids=ids,attention_mask=torch.ones_like(ids),logits_to_keep=1,use_cache=False).logits[0,0].float()
                assert logits.numel() == 248320 and torch.isfinite(logits).all()
                log_p = (logits/2.0).log_softmax(-1)
                probabilities[i].copy_(log_p.cpu())
                generated = model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
                    max_new_tokens=8,do_sample=False,use_cache=False,pad_token_id=tokenizer.eos_token_id)
            output = generated[0,len(prefix):]
            text = tokenizer.decode(output,skip_special_tokens=True)
            target = pilot.parse_answer(text,row)
            eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
            eos = {eos} if isinstance(eos,int) else set(eos)
            cutoff = output.numel() == 8 and int(output[-1]) not in eos
            if target is None or cutoff:
                pilot.write_json(folder/"invalid_response.json",{"source_id":row["source_id"],"response":text,"cutoff":cutoff})
                raise ValueError("Teacher format error; no filtering or gold repair allowed")
            assert int(logits.argmax()) == tokenizer.encode(target,add_special_tokens=False)[0]
            choice_ids = [tokenizer.encode(c["label"],add_special_tokens=False)[0] for c in row["choices"]]
            choice_probs = logits[choice_ids].softmax(-1).cpu().tolist()
            responses.append({**row,"teacher_target":target,"response":text,
                "correct":target == row["gold"],"choice_probabilities_T1":dict(zip([c["label"] for c in row["choices"]],choice_probs))})
            if (i+1)%50 == 0 or i+1 == len(rows):
                print(f"full Teacher targets {i+1}/{len(rows)}",flush=True)
                metadata["completed"] = i+1
                pilot.write_json(summary_path,metadata)
        assert torch.isfinite(probabilities).all() and torch.allclose(probabilities.logsumexp(-1),torch.zeros(len(rows)),atol=1e-5)
        save_file({"answer_log_probs":probabilities.contiguous()},str(target_path))
        pilot.write_rows(folder/"teacher_responses.jsonl",responses)
        metadata.update(loading)
        metadata.update({"status":"passed","extraction_seconds":perf_counter()-started,
            "prefix_sha256":prefix_hashes,"output_sha256":sha(target_path),
            "output_bytes":target_path.stat().st_size,"response_sha256":sha(folder/"teacher_responses.jsonl"),
            "teacher_gold_agreement":sum(r["correct"] for r in responses)/len(rows),
            "teacher_wrong":sum(not r["correct"] for r in responses),
            "peak_allocated_bytes":torch.cuda.max_memory_allocated()})
    except Exception as exc:
        metadata.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        pilot.write_json(summary_path,metadata)


def context(scope, condition, seed, *, training=False):
    import torch
    from safetensors.torch import load_file
    if training:
        _, protocol_hash = require_protocol()
    else:
        protocol_hash = None
    config,_ = trainer.run_config("logit_kd" if condition in SOFT else condition, seed)
    variant = condition if condition in NEW else "forward"
    if condition in SOFT:
        config["distillation"]["variant"] = variant
        config["loss"] = f"0.5 * (T^2 {variant} answer divergence + shared EOS CE)"
    config["dataset_scope"] = scope
    config["partial_batch_normalization"] = "divide each sample by actual accumulation-group size"
    if scope == "pilot":
        rows,teacher = trainer.load_training_rows()
        examples_summary = json.loads((logit_kd.OUTPUT/"soft_targets_summary.json").read_text(encoding="utf-8"))
        cache_path = logit_kd.OUTPUT/"teacher_log_probs.safetensors"
        assert sha(cache_path) == examples_summary["output_sha256"]
        cache_hash = examples_summary["output_sha256"]
        target_ids = examples_summary["sample_ids"]
        prefix_hashes = examples_summary["prefix_sha256"]
    else:
        rows,path = rows_for(scope)
        summary = json.loads((OUT/"targets/full/summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "passed" and summary["data_sha256"] == sha(path)
        response_path = OUT/"targets/full/teacher_responses.jsonl"
        assert sha(response_path) == summary["response_sha256"]
        responses = pilot.read_rows(response_path)
        assert [r["source_id"] for r in responses] == [r["source_id"] for r in rows]
        for row,response in zip(rows,responses):
            assert all(response[k] == v for k,v in row.items())
            row["teacher_target"] = response["teacher_target"]
        teacher = {"data_sha256":sha(path),"output_sha256":summary["response_sha256"]}
        cache_path = OUT/"targets/full/teacher_log_probs.safetensors"
        cache_hash = summary["output_sha256"]
        assert sha(cache_path) == cache_hash
        target_ids,prefix_hashes = summary["sample_ids"],summary["prefix_sha256"]
    targets = None
    if condition in SOFT:
        targets = load_file(str(cache_path),device="cpu")["answer_log_probs"]
        assert targets.shape == (len(rows),248320) and targets.dtype == torch.float32
        assert torch.isfinite(targets).all() and torch.allclose(targets.logsumexp(-1),torch.zeros(len(rows)),atol=1e-5)
    return {"scope":scope,"rows":rows,"teacher":teacher,"config":config,"targets":targets,
        "target_ids":target_ids,"prefix_sha256":prefix_hashes,"soft_targets_sha256":cache_hash,
        "probe_path":OUT/"probes"/scope/f"{condition}.json","protocol_sha256":protocol_hash}


def probe(scope,condition):
    import torch
    ctx = context(scope,condition,42)
    path = ctx["probe_path"]
    if ready(path):
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["config"] == ctx["config"] and saved["code_sha256"] == code_hashes()
        return
    metadata = {"status":"running","config":ctx["config"],"code_sha256":code_hashes(),"probe_model_reused":False}
    pilot.write_json(path,metadata)
    try:
        model,tokenizer,parameters,optimizer = trainer.load_student(42)
        examples = [trainer.encode_example(r,condition,tokenizer) for r in ctx["rows"]]
        assert ctx["target_ids"] == [e["source_id"] for e in examples]
        assert ctx["prefix_sha256"] == [logit_kd.prefix_hash(e) for e in examples]
        index = max(range(len(examples)),key=lambda i:len(examples[i]["input_ids"]))
        initial = trainer.adapter_hash(model)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(8):
            signal = {} if ctx["targets"] is not None else None
            loss = trainer.answer_loss(model,examples[index],None if ctx["targets"] is None else ctx["targets"][index],
                                       condition if condition in NEW else "forward",signal)
            (loss/8).backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        assert trainer.adapter_hash(model) != initial and model.get_input_embeddings().weight.grad is None
        metadata.update(status="passed",source_id=examples[index]["source_id"],
            sequence_tokens=len(examples[index]["input_ids"]),loss=loss.item(),grad_norm=float(norm),
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:
        metadata.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        pilot.write_json(path,metadata)
    print(f"GPU probe passed: {scope} {condition}",flush=True)


def freeze_training():
    path = OUT/"training_protocol.json"
    if ready(path):
        require_protocol()
        return
    assert ready(OUT/"self_check.json")
    assert json.loads((OUT/"self_check.json").read_text(encoding="utf-8"))["code_sha256"] == code_hashes()
    for scope in ("full",):
        for condition in CONDITIONS:
            probe_path = OUT/"probes"/scope/f"{condition}.json"
            assert ready(probe_path)
            assert json.loads(probe_path.read_text(encoding="utf-8"))["code_sha256"] == code_hashes()
    payload = {"status":"passed","frozen_at_kst":trainer.now(),"code_sha256":code_hashes(),
        "conditions":CONDITIONS,"seeds":SEEDS,"scopes":{"full":1118},
        "epoch_diagnostics":{"seed":42,"epochs":[0,1,2,3],"n_per_split":MONITOR_N,
            "selection":"lowest SHA256(source_id), fixed per split and condition",
            "sample_ids":{split:[r["source_id"] for r in monitor_rows("full",split)] for split in ("train","validation")},
            "used_for_checkpoint_selection":False},
        "main_metrics":["generation_accuracy","choice_nll","paired_95ci"],
        "temperature":2.0,"vocabulary_size":248320,"fixed_mix_forward_weight":0.5,
        "gate":"hard 1[1-H(P)/ln(V) >= 1-H(Q)/ln(V)], teacher eval, student same train forward, no gate gradient",
        "checkpoint":"fixed epoch 3","teacher_response_filter":"none",
        "primary_comparison":"adaptive_mix minus fixed_mix generation accuracy; full train is main experiment",
        "secondary_metrics":["brier","choice_nll","ece","teacher_error_patterns"],"bootstrap_replicates":5000,
        "bootstrap_seed":42,"ci_scope":"paired questions after averaging the three observed training seeds",
        "versions":{p:importlib.metadata.version(p) for p in ("torch","transformers","peft","bitsandbytes","numpy","safetensors")},
        "data_sha256":json.loads((pilot.OUTPUT/"data_manifest.json").read_text(encoding="utf-8"))["export_sha256"],
        "full_target_summary_sha256":sha(OUT/"targets/full/summary.json"),
        "pilot_target_summary_sha256":sha(logit_kd.OUTPUT/"soft_targets_summary.json"),
        "test_evaluation":False}
    snapshot = OUT/"sources_frozen"
    snapshot.mkdir(exist_ok=True)
    for name,checksum in payload["code_sha256"].items():
        destination = snapshot/name
        if destination.exists():
            assert sha(destination) == checksum
        else:
            destination.write_bytes((ROOT/name).read_bytes())
    pilot.write_json(path,payload)


def train(scope,condition,seed):
    protocol,_ = require_protocol()
    if scope not in protocol["scopes"]:
        raise ValueError("Additional pilot training was removed from the approved experiment")
    run = run_path(scope,condition,seed)
    if ready(run/"summary.json"):
        return
    ctx = context(scope,condition,seed,training=True)
    def monitor(model,tokenizer,epoch,step,folder):
        before = trainer.adapter_hash(model)
        summaries = {}
        with diagnostic.evaluation_scope(model):
            for split in ("train","validation"):
                rows = monitor_rows(scope,split)
                records,summaries[split] = diagnostic.evaluate(model,tokenizer,rows,split,epoch)
                pilot.write_rows(folder/f"epoch_{epoch}_{split}.jsonl",records)
        assert trainer.adapter_hash(model) == before
        pilot.write_json(folder/f"epoch_{epoch}.json",{"status":"passed","epoch":epoch,
            "optimizer_step":step,"seed":seed,"scope":scope,"metrics":summaries,
            "weights_unchanged":True,"validation_used_for_selection":False,"test_evaluation":False,
            "diagnostic_subset":True,"n_per_split":MONITOR_N})
    trainer.train(condition,seed,output_dir=run,monitor=monitor if seed == 42 else None,experiment=ctx)


def freeze_test():
    require_protocol()
    path = OUT/"test_manifest.json"
    if ready(path):
        assert json.loads(path.read_text(encoding="utf-8"))["code_sha256"] == code_hashes()
        return
    models = {"teacher_baseline":{},"student_baseline":{}}
    for scope in ("full",):
        for seed in SEEDS:
            summaries = [json.loads((run_path(scope,c,seed)/"summary.json").read_text(encoding="utf-8")) for c in CONDITIONS]
            assert len({s["initial_adapter_sha256"] for s in summaries}) == 1
            assert len({tuple(s["epoch_order_sha256"]) for s in summaries}) == 1
            assert all(s["train_rows"] == (256 if scope == "pilot" else 1118) for s in summaries)
            assert all(s["optimizer_steps_completed"] == (96 if scope == "pilot" else 420) for s in summaries)
            assert all(not s["validation_used_for_training"] for s in summaries)
        for condition in CONDITIONS:
            for seed in SEEDS:
                run = run_path(scope,condition,seed)
                assert ready(run/"summary.json")
                assert ready(OUT/"scores/validation"/f"{scope}_{condition}_{seed}_summary.json")
                summary = json.loads((run/"summary.json").read_text(encoding="utf-8"))
                models[f"{scope}/{condition}/{seed}"] = {str(run/"adapter"/name):checksum
                    for name,checksum in summary["adapter_files_sha256"].items()}
    pilot.write_json(path,{"status":"passed","frozen_at_kst":trainer.now(),"code_sha256":code_hashes(),
        "training_protocol_sha256":sha(OUT/"training_protocol.json"),"models":models,
        "test_data_sha256":sha(pilot.DATA/"test.jsonl"),"test_evaluation_before_freeze":False,
        "no_post_test_selection_or_retraining":True})


def self_check():
    import torch
    import torch.nn.functional as F
    torch.manual_seed(23)
    targets = torch.tensor([0,1])
    t = torch.tensor([1.2,-0.4,0.7])
    p = (t/2).log_softmax(-1)
    x = torch.tensor([[.1,1.4,-.6],[.3,-.8,.2]],requires_grad=True)
    q = (x[0]/2).log_softmax(-1)
    f = (p.exp()*(p-q)).sum()
    r = (q.exp()*(q-p)).sum()
    eos = F.cross_entropy(x[1:],targets[1:])
    for condition,expected in (("forward",f),("reverse_kl",r),("fixed_mix",(f+r)/2)):
        actual = logit_kd.distillation_loss(x,targets,p,condition)
        direct = .5*(4*expected+eos)
        assert torch.allclose(actual,direct,atol=1e-6)
        assert torch.allclose(torch.autograd.grad(actual,x,retain_graph=True)[0],
                              torch.autograd.grad(direct,x,retain_graph=True)[0],atol=1e-6)
        changed = targets.clone();changed[0]=2
        assert torch.allclose(actual,logit_kd.distillation_loss(x,changed,p,condition))
    sharp = torch.tensor([4.,0.,-2.]).log_softmax(-1)
    flat = torch.zeros(3).log_softmax(-1)
    assert logit_kd.confidence_gate(sharp,flat)[2].item() == 1
    assert logit_kd.confidence_gate(flat,sharp)[2].item() == 0
    assert logit_kd.confidence_gate(sharp,sharp)[2].item() == 1
    assert not logit_kd.confidence_gate(p,q)[2].requires_grad
    for teacher,student,branch in ((sharp,flat,"forward"),(flat,sharp,"reverse_kl")):
        logits = torch.stack([2*student,x[1].detach()]).requires_grad_()
        a = logit_kd.distillation_loss(logits,targets,teacher,"adaptive_mix")
        b = logit_kd.distillation_loss(logits,targets,teacher,branch)
        assert torch.allclose(a,b,atol=1e-6)
        a.backward()
        assert torch.isfinite(logits.grad).all() and teacher.grad is None
    for mode in ("forward","reverse_kl","fixed_mix","adaptive_mix"):
        same = torch.stack([t,x[1].detach()])
        loss = logit_kd.distillation_loss(same,targets,p,mode)
        assert abs((loss-.5*eos).item()) < 1e-6
    for count in (3,4,5):
        labels = list("ABCDE"[:count])
        record = {"gold":labels[0],"choice_predicted":labels[0],"choice_confidence":1/count,
                  "choice_probabilities":dict.fromkeys(labels,1/count),
                  "choice_log_probabilities":dict.fromkeys(labels,-math.log(count))}
        scores = diagnostic.calibration([record])
        assert abs(scores["multiclass_brier"]-(1-1/count)) < 1e-9
        assert abs(scores["choice_nll"]-math.log(count)) < 1e-9
    # The last six of 1118 samples form their own accumulation group.
    groups = [min(8,1118-start) for start in range(0,1118,8)]
    assert len(groups) == 140 and groups[-1] == 6 and sum(groups) == 1118
    values = torch.arange(1.,7.,requires_grad=True)
    accumulated = sum(v/6 for v in values)
    assert torch.allclose(accumulated,values.mean())
    assert torch.allclose(torch.autograd.grad(accumulated,values)[0],torch.full((6,),1/6))
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    with TemporaryDirectory() as directory:
        with patch(__name__+".OUT",Path(directory)), patch(__name__+".require_protocol",return_value=({"scopes":{"full":1118}},None)):
            try:
                check_test("full","student",None,42)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("Unfrozen test evaluation unexpectedly allowed")
    jobs = pipeline_steps()
    planned = {(c,str(seed)) for c in CONDITIONS for seed in SEEDS}
    train_jobs = [j for j in jobs if j[0] == "train"]
    assert len(train_jobs) == 18 and {(j[4],str(j[6])) for j in train_jobs} == planned
    assert all(j[2] == "full" for j in train_jobs) and all("pilot" not in j for j in jobs)
    test_jobs = [j for j in jobs if j[0] == "score" and "test" in j]
    assert len(test_jobs) == 20
    assert jobs.index(("freeze-test",)) > max(i for i,j in enumerate(jobs) if j[0] == "train")
    for split in ("train","validation"):
        sample = monitor_rows("full",split)
        assert len(sample) == MONITOR_N and len({r["source_id"] for r in sample}) == MONITOR_N
        assert sample == monitor_rows("full",split) and all(r["source_split"] == split for r in sample)
    with patch(__name__+".require_protocol",return_value=({"scopes":{"full":1118}},None)):
        try:
            train("pilot","gold_sft",42)
        except ValueError:
            pass
        else:
            raise AssertionError("Removed pilot training unexpectedly allowed")
    import analyze_kl
    analyze_kl.self_check()
    trainer.self_check()
    diagnostic.self_check()
    logit_kd.self_check()
    pilot.write_json(OUT/"self_check.json",{"status":"passed","code_sha256":code_hashes(),
        "checks":["KL values/gradients","hard gate branches/ties/no gradient","no hard/gold answer in KL",
                  "3/4/5 choice calibration","last partial group normalization","unfrozen test guard",
                  "full-only 18 training/20 test jobs","fixed 64-item diagnostic subsets",
                  "removed pilot training blocked","existing causal loss/RNG/calibration checks"],"test_evaluation":False})
    print("EXTENSION SELF-CHECK PASS",flush=True)


def child(*args):
    command = [sys.executable,"-B","-X","utf8",str(Path(__file__).resolve()),*map(str,args)]
    pilot.write_json(OUT/"status.json",{"status":"running","current_command":args,"updated_at_kst":trainer.now(),
                                     "test_phase":bool("test" in args)})
    subprocess.run(command,check=True,cwd=ROOT)


def pipeline_steps():
    steps = []
    steps.append(("self-check",))
    for role in ("teacher","student"):
        steps.append(("score","--role",role))
    steps.append(("prepare-full",))
    for scope in ("full",):
        for condition in CONDITIONS:
            steps.append(("probe","--scope",scope,"--condition",condition))
    steps.append(("freeze-training",))
    for scope in ("full",):
        for condition in CONDITIONS:
            for seed in SEEDS:
                steps.append(("train","--scope",scope,"--condition",condition,"--seed",seed))
                steps.append(("score","--scope",scope,"--condition",condition,"--seed",seed))
        steps.append(("analyze","--scope",scope))
    steps.append(("freeze-test",))
    for role in ("teacher","student"):
        steps.append(("score","--role",role,"--split","test"))
    for scope in ("full",):
        for condition in CONDITIONS:
            for seed in SEEDS:
                steps.append(("score","--scope",scope,"--condition",condition,"--seed",seed,"--split","test"))
        steps.append(("analyze","--scope",scope,"--split","test"))
    return steps


def pipeline():
    steps = pipeline_steps()
    for number,args in enumerate(steps,1):
        print(f"PIPELINE {number}/{len(steps)}: {' '.join(map(str,args))}",flush=True)
        try:
            child(*args)
        except Exception as exc:
            pilot.write_json(OUT/"status.json",{"status":"failed","failed_job":number,"current_command":args,
                "error":f"{type(exc).__name__}: {exc}","updated_at_kst":trainer.now()})
            raise
    preserve = json.loads((OUT/"preservation.json").read_text(encoding="utf-8"))["original_artifacts_sha256"]
    changed = [name for name,checksum in preserve.items() if sha(ROOT/name) != checksum]
    assert not changed,changed
    scope_record = json.loads((OUT/"scope_change.json").read_text(encoding="utf-8"))
    completed_pilot = scope_record["completed_extension_pilot_artifacts_sha256"]
    assert all(sha(ROOT/name) == checksum for name,checksum in completed_pilot.items())
    pilot.write_json(OUT/"status.json",{"status":"passed","completed_jobs":len(steps),
        "experiment_scope":"full","completed_training_runs":18,"pilot_artifacts_preserved":len(completed_pilot),
        "updated_at_kst":trainer.now(),"original_artifacts_preserved":len(preserve),"test_evaluation":True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("self-check","score","prepare-full","probe","freeze-training","train","freeze-test","analyze","analyze-baselines","pipeline"))
    parser.add_argument("--scope",choices=("pilot","full"),default="full")
    parser.add_argument("--condition",choices=CONDITIONS)
    parser.add_argument("--seed",type=int,choices=SEEDS,default=42)
    parser.add_argument("--role",choices=("teacher","student"),default="student")
    parser.add_argument("--split",choices=("validation","test"),default="validation")
    args = parser.parse_args()
    if args.action == "score": score(args.scope,args.role,args.condition,args.seed,args.split)
    elif args.action == "probe":
        assert args.condition;probe(args.scope,args.condition)
    elif args.action == "train":
        assert args.condition;train(args.scope,args.condition,args.seed)
    elif args.action in ("analyze","analyze-baselines"):
        import analyze_kl
        analyze_kl.analyze(args.scope,args.split,baseline_only=args.action == "analyze-baselines")
    else:
        {"self-check":self_check,"prepare-full":prepare_full,"freeze-training":freeze_training,
         "freeze-test":freeze_test,"pipeline":pipeline}[args.action]()
