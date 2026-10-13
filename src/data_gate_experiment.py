"""One-seed external-data gate comparison; frozen ARC artifacts are never modified."""
import argparse
import gc
import importlib.metadata
import json
import random
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
import gate_continuity as data
import gate_experiment as gate
import logit_kd
import monitor_training as diagnostic
import pilot
import soft_gate
import train_student as trainer
from bayesian_analysis import sha, read, rows, write
from choice_kd import require_preservation
from teacher_quantization import ROOT, CACHE, MODEL_ID, REVISION, load_text_model

OUT = ROOT / 'outputs/data_gates_v1'
PUBLIC = ROOT / 'results/data_gates_v1'
DATA = ROOT / 'data/data_gates_v1'
PROTOCOL = PUBLIC / 'protocol.json'
METHODS = ('fixed_mix', 'adaptive_mix', 'adaptive_soft')
DATASETS = ('openbookqa', 'hellaswag')
CONFIG = {**trainer.CONFIG, 'loss':'0.5 * (T^2 * gated Forward/Reverse KL + shared EOS CE)'}


def save_rows(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(str(path))
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in records),
                    encoding='utf-8', newline='\n')


def pick_heldout(items, n=500):
    selected, questions, scenes = [], set(), set()
    for item in random.Random(42).sample(items, len(items)):
        q, scene = item['question_sha256'], item.get('scene_id')
        if q in questions or (scene is not None and scene in scenes):
            continue
        selected.append(item)
        questions.add(q)
        if scene is not None:
            scenes.add(scene)
        if len(selected)==2*n:
            return selected[:n], selected[n:]
    raise ValueError('Insufficient disjoint question/scene groups')


def clean_train(items, validation, conflicting_ids):
    qval = {r['question_sha256'] for r in validation}
    kept = {}
    for item in items:
        if item['question_sha256'] not in qval and item['source_id'] not in conflicting_ids:
            kept.setdefault(item['question_sha256'], item)
    return list(kept.values())


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(pilot.STUDENT_ID, revision=pilot.STUDENT_REVISION,
        cache_dir=str(CACHE), local_files_only=True, trust_remote_code=False)


def encode(item, tok):
    rendered = tok.apply_chat_template([{'role':'user','content':data.screen_prompt(item)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefix = tok.encode(rendered, add_special_tokens=False)
    labels = [c['label'] for c in item['choices']]
    choice_ids = [tok.encode(c, add_special_tokens=False) for c in labels]
    if labels!=list('ABCD') or not prefix:
        raise ValueError('Four original-order choices required')
    assert all(len(ids)==1 and tok.encode(rendered+label, add_special_tokens=False)==prefix+ids
               for label,ids in zip(labels,choice_ids))
    assert len(prefix)+2<=trainer.CONFIG['max_sequence_length'], 'Do not truncate or resample silently'
    target = item.get('teacher_target', item['gold'])
    ids = prefix+[choice_ids[labels.index(target)][0],tok.eos_token_id]
    return {'source_id':item['source_id'], 'prefix':prefix, 'choice_ids':[x[0] for x in choice_ids],
        'input_ids':ids, 'prediction_positions':[len(prefix)-1,len(prefix)],
        'target_ids':ids[-2:], 'prompt_tokens':len(prefix)}


def require():
    p = read(PROTOCOL)
    assert p['versions']=={k:importlib.metadata.version(k) for k in p['versions']}
    for name,h in p['source_sha256'].items():
        assert sha(ROOT/'src'/name)==h,name
    for name,h in p['data_sha256'].items():
        assert sha(ROOT/name)==h,name
    assert p['config']==CONFIG and logit_kd.TEMPERATURE==2 and soft_gate.MARGIN==.05
    return p


def prepare():
    if PROTOCOL.exists():
        raise FileExistsError('Existing frozen protocol preserved')
    preservation = require_preservation()
    previous = read(data.PROTOCOL)
    for name,h in previous['data_sha256'].items():
        assert sha(ROOT/name)==h,name
    tok = tokenizer()
    detail, hashes = {}, {}
    for name in DATASETS:
        raw_train = rows(data.DATA/name/'train.jsonl')
        validation = rows(data.DATA/name/'validation.jsonl')
        conflict = set(read(data.PUBLIC/'datasets.json')[name]['conflicting_gold_source_ids'])
        eligible = clean_train(raw_train,validation,conflict)
        train = random.Random(42).sample(eligible,1118)
        dev, holdout = pick_heldout(validation) if name=='hellaswag' else (validation,None)
        subsets = {'train':train,'validation':dev}
        if holdout is not None:
            subsets['holdout']=holdout
        lengths = {}
        for split, records in subsets.items():
            examples = [encode(r,tok) for r in records]
            lengths[split] = data.describe([len(e['input_ids']) for e in examples])
            path=DATA/name/f'{split}.jsonl'
            save_rows(path,records)
            hashes[str(path.relative_to(ROOT))]=sha(path)
        assert not ({r['question_sha256'] for r in train}&{r['question_sha256'] for r in dev})
        detail[name]={'train_n':1118,'validation_n':len(dev),'eligible_unique_train_questions':len(eligible),
            'token_lengths_including_two_targets':lengths,
            'gradient_probe_ids':[r['source_id'] for r in train[:8]],
            'final_evaluation':'official test, input-only frozen exclusions applied after training' if name=='openbookqa'
                else '500 untouched validation items; no shared question or scene with development set',
            'dataset_revision':data.SPECS[name]['revision']}
    sources = dict(read(ROOT/'results/protocols/gate_test_manifest.json')['code_sha256'])
    for name in ('data_gate_experiment.py','report_data_gates.py','gate_continuity.py','choice_kd.py',
                 'bayesian_analysis.py'):
        sources[name]=sha(ROOT/'src'/name)
    p={'version':'data_gates_v1','frozen_at_kst':trainer.now(),'seed':42,'datasets':detail,
       'config':CONFIG,'methods':list(METHODS),'source_sha256':sources,'data_sha256':hashes,
       'teacher':{'id':MODEL_ID,'revision':REVISION,'quantization':'NF4 double quant, BF16 compute'},
       'train_budget_basis':'Inherited ARC 1118 items/3 epochs; one seed is exploratory compute budget',
       'evaluation':'T=1 valid-letter argmax/NLL, common neutral prompt; not official completion scoring',
       'teacher_hard_target':'valid-letter argmax for common EOS conditioning; full vocabulary soft targets at T=2',
       'soft_formula':'sigmoid(c_teacher-c_student-0.05), selection gradient retained',
       'checkpoint':'fixed epoch 3; no validation or final-holdout selection',
       'bootstrap':{'replicates':5000,'seed':42,'pairing':'same unique question across methods, one training seed'},
       'training_order':[['openbookqa',m] for m in METHODS]+[['hellaswag',m] for m in reversed(METHODS)],
       'openbookqa_final_rule':'no test scoring before all six adapters are frozen; exclude train/dev question hashes, keep first unique test question, reject over-budget inputs without truncation',
       'preservation':preservation,'versions':{k:importlib.metadata.version(k) for k in
                 ('torch','transformers','peft','bitsandbytes','safetensors')},
       'sources':{'gate':'https://arxiv.org/html/2609.36734v1','hellaswag_test_policy':'https://rowanzellers.com/hellaswag/'}}
    write(PROTOCOL,p)
    write(PUBLIC/'status.json',{'stage':'prepared','training_complete':0,'training_planned':6,'final_evaluation_started':False})
    print('Prepared 2 datasets: 1118 train, 500 dev each; held-out evaluation sealed until six trainings finish.',flush=True)


def load_model(role,adapter=None):
    from transformers import BitsAndBytesConfig
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16) if role=='teacher' else None
    model,_=load_text_model(MODEL_ID if role=='teacher' else pilot.STUDENT_ID,
                           REVISION if role=='teacher' else pilot.STUDENT_REVISION,quant)
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,adapter,is_trainable=False)
    model.eval().requires_grad_(False)
    return model


def evaluate(model, items, tok, path, label):
    if path.exists():
        old=read(path)
        assert old['protocol_sha256']==sha(PROTOCOL)
        assert [r['source_id'] for r in old['records']]==[r['source_id'] for r in items]
        return old
    records=[]
    with diagnostic.evaluation_scope(model):
        for i,item in enumerate(items):
            e=encode(item,tok)
            logits=model(input_ids=torch.tensor([e['prefix']],device='cuda'),logits_to_keep=1,
                         use_cache=False).logits[0,-1].float()
            scores=logits[e['choice_ids']]
            logp=scores.log_softmax(-1).cpu()
            assert torch.isfinite(logp).all() and torch.isfinite(logits).all()
            predicted=item['choices'][int(scores.argmax())]['label']
            full_logp=(logits/2).log_softmax(-1)
            confidence=float(1+(full_logp.exp()*full_logp).sum()/np.log(logits.numel()))
            assert -.00001<=confidence<=1.00001
            records.append({'source_id':item['source_id'],'question_sha256':item['question_sha256'],
                'gold':item['gold'],'predicted':predicted,'correct':predicted==item['gold'],
                'choice_log_probabilities':logp.tolist(), 'prefix_tokens':len(e['prefix']),
                'full_vocab_confidence_T2':confidence,
                'full_top1_is_valid_choice':int(logits.argmax()) in e['choice_ids'],
                'choice_mass_T1':float(logits.softmax(-1)[e['choice_ids']].sum())})
            if (i+1)%100==0:
                print(f'{label}: {i+1}/{len(items)}',flush=True)
    metric={'n':len(records),'accuracy':float(np.mean([r['correct'] for r in records])),
            'choice_nll':float(np.mean([-r['choice_log_probabilities'][ord(r['gold'])-65] for r in records]))}
    value={'protocol_sha256':sha(PROTOCOL),'label':label,'metrics':metric,'records':records}
    write(path,value)
    print(f'{label}: accuracy={metric["accuracy"]:.2%}, NLL={metric["choice_nll"]:.4f}',flush=True)
    return value


def teacher_targets(name):
    from safetensors.torch import save_file
    require()
    folder=OUT/name/'targets'; summary=folder/'summary.json'
    if summary.exists():
        v=read(summary)
        if v['status']=='passed':
            for filename,h in v['files_sha256'].items(): assert sha(folder/filename)==h,filename
            return
        raise FileExistsError('Incomplete Teacher extraction preserved')
    folder.mkdir(parents=True,exist_ok=False)
    metadata={'status':'running','protocol_sha256':sha(PROTOCOL)}
    write(summary,metadata)
    tok=tokenizer(); items=rows(DATA/name/'train.jsonl'); model=load_model('teacher')
    from transformers import AutoTokenizer
    teacher_tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION,cache_dir=str(CACHE),
                         local_files_only=True,trust_remote_code=False)
    assert tok.get_vocab()==teacher_tok.get_vocab() and tok.special_tokens_map==teacher_tok.special_tokens_map
    stored=torch.empty((len(items),logit_kd.VOCAB_SIZE),dtype=torch.float32)
    encoded=[encode(r,tok) for r in items]
    assert [encode(r,teacher_tok)['prefix'] for r in items]==[e['prefix'] for e in encoded]
    try:
        with torch.inference_mode():
            for i,(item,e) in enumerate(zip(items,encoded)):
                logits=model(input_ids=torch.tensor([e['prefix']],device='cuda'),logits_to_keep=1,
                             use_cache=False).logits[0,-1].float()
                logp=(logits/2).log_softmax(-1)
                assert logits.numel()==logit_kd.VOCAB_SIZE and torch.isfinite(logp).all()
                stored[i].copy_(logp.cpu())
                item['teacher_target']='ABCD'[int(logits[e['choice_ids']].argmax())]
                item['teacher_full_top1_is_valid_choice']=int(logits.argmax()) in e['choice_ids']
                if (i+1)%64==0:
                    write(summary,{**metadata,'completed':i+1})
                    print(f'{name} Teacher targets: {i+1}/{len(items)}',flush=True)
        assert torch.allclose(stored.logsumexp(-1),torch.zeros(len(items)),atol=1e-5)
        save_file({'answer_log_probs':stored},str(folder/'teacher_log_probs.safetensors'))
        save_rows(folder/'train_targets.jsonl',items)
        metadata.update(status='passed',n=len(items),teacher_train_accuracy=sum(r['teacher_target']==r['gold'] for r in items)/len(items),
            full_top1_invalid_choice_count=sum(not r['teacher_full_top1_is_valid_choice'] for r in items),
            files_sha256={f:sha(folder/f) for f in ['teacher_log_probs.safetensors','train_targets.jsonl']})
        evaluate(model,rows(DATA/name/'validation.jsonl'),tok,OUT/name/'validation/teacher.json',name+' Teacher dev')
    except BaseException as exc:
        metadata.update(status='failed',error=str(exc)); raise
    finally:
        write(summary,metadata)


def targets(name):
    from safetensors.torch import load_file
    folder=OUT/name/'targets'; v=read(folder/'summary.json')
    assert v['status']=='passed' and v['protocol_sha256']==sha(PROTOCOL)
    for f,h in v['files_sha256'].items(): assert sha(folder/f)==h,f
    items=rows(folder/'train_targets.jsonl'); values=load_file(folder/'teacher_log_probs.safetensors')['answer_log_probs']
    assert [r['source_id'] for r in items]==[r['source_id'] for r in rows(DATA/name/'train.jsonl')]
    assert values.shape==(1118,logit_kd.VOCAB_SIZE) and torch.isfinite(values).all()
    return items,values


def gradient_audit(model, parameters, items, values, tok):
    before=trainer.adapter_hash(model); was_training=model.training; model.eval()
    records=[]
    try:
        for i,item in enumerate(items[:8]):
            e=encode(item,tok)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(input_ids=torch.tensor([e['input_ids']],device='cuda'),use_cache=False,
                    logits_to_keep=torch.tensor(e['prediction_positions'],device='cuda')).logits[0].float()
            p=values[i].cuda(); q=(logits[0]/2).log_softmax(-1)
            f=F.kl_div(q,p,reduction='sum',log_target=True); r=(q.exp()*(q-p)).sum()
            cp,cq,_=logit_kd.confidence_gate(p,q); w=torch.sigmoid(cp-cq-.05)
            learning=2*(w.detach()*f+(1-w.detach())*r)
            selection=2*(f.detach()-r.detach())*w
            eos=.5*F.cross_entropy(logits[1:],torch.tensor(e['target_ids'][1:],device='cuda'))
            complete=soft_gate.distillation_loss(logits,torch.tensor(e['target_ids'],device='cuda'),p)
            logit_parts=[torch.autograd.grad(x,logits,retain_graph=True)[0] for x in (learning,selection,eos)]
            logit_total=torch.autograd.grad(complete,logits,retain_graph=True)[0]
            assert torch.allclose(logit_total,sum(logit_parts),atol=1e-6,rtol=1e-4)
            parts=[torch.autograd.grad(x,parameters,retain_graph=True,allow_unused=True) for x in (learning,selection,eos)]
            total=torch.autograd.grad(complete,parameters,allow_unused=True)
            norms=[]
            for part in parts:
                norms.append(float(torch.sqrt(sum(x.float().square().sum() for x in part if x is not None))))
            error_sq=total_sq=torch.zeros((),device='cuda')
            for j,t in enumerate(total):
                if t is not None:
                    combined=sum(part[j] for part in parts if part[j] is not None)
                    assert torch.isfinite(t).all()
                    error_sq=error_sq+(t-combined).float().square().sum()
                    total_sq=total_sq+t.float().square().sum()
            relative_error=float((error_sq/total_sq.clamp_min(1e-30)).sqrt())
            assert np.isfinite(relative_error)
            rounding_warning=relative_error>4*torch.finfo(torch.bfloat16).eps
            records.append({'source_id':item['source_id'],'forward_weight':float(w.detach()),
                'learning_grad_norm':norms[0],'selection_grad_norm':norms[1],'eos_grad_norm':norms[2],
                'selection_to_learning_ratio':norms[1]/max(norms[0],1e-30),
                'parameter_decomposition_relative_error':relative_error, 'bf16_rounding_warning':rounding_warning})
            del parts,total,logits,learning,selection,eos,f,r,p,q,cp,cq,w,complete,logit_parts,logit_total
    finally:
        model.train(was_training)
    assert trainer.adapter_hash(model)==before and all(p.grad is None for p in parameters)
    return {'n':8,'scope':'Deterministic eval-mode train items; unclipped LoRA gradient norms, no optimizer update',
            'records':records,'fp32_logit_decomposition_verified':True,'bf16_parameter_gradients_measured':True,
            'parameter_relative_error_warning_threshold':4*torch.finfo(torch.bfloat16).eps,
            'tolerance_basis':'BF16 backward rounds before component sums; FP32 logit identity checked separately'}


def probe(name):
    require(); path=OUT/name/'probe.json'
    if path.exists():
        assert read(path)['status']=='passed'; return
    items,values=targets(name); model,tok,parameters,optimizer=trainer.load_student(42)
    initial=trainer.adapter_hash(model); peak=0
    audit=gradient_audit(model,parameters,items,values,tok)
    for method in METHODS:
        optimizer.zero_grad(set_to_none=True)
        for i in range(8):
            loss=gate.loss(model,encode(items[i],tok),values[i],method,{})
            (loss/8).backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        assert float(norm)>0
        peak=max(peak,torch.cuda.max_memory_allocated())
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    assert trainer.adapter_hash(model)!=initial and model.get_input_embeddings().weight.grad is None
    write(path,{'status':'passed','protocol_sha256':sha(PROTOCOL),'initial_adapter_sha256':initial,
                'three_loss_gradients_finite':True,'one_disposable_optimizer_update':True,
                'peak_allocated_bytes':peak,'gradient_audit':audit})
    print(name+' disposable GPU/gradient probe PASSED',flush=True)


def train(name,method):
    if method not in METHODS: raise ValueError('Unsupported training condition')
    require(); run=OUT/name/'training'/method; summary=run/'summary.json'
    if summary.exists():
        done=read(summary)
        if done['status']=='passed':
            assert done['protocol_sha256']==sha(PROTOCOL) and done['optimizer_steps_completed']==420
            for f,h in done['adapter_files_sha256'].items(): assert sha(run/'adapter'/f)==h,f
            return
        raise FileExistsError('Incomplete training preserved; diagnose before recovery')
    probe_data=read(OUT/name/'probe.json'); assert probe_data['status']=='passed'
    items,values=targets(name); model,tok,parameters,optimizer=trainer.load_student(42)
    initial=trainer.adapter_hash(model); assert initial==probe_data['initial_adapter_sha256']
    examples=[encode(r,tok) for r in items]; dev=rows(DATA/name/'validation.jsonl')
    from transformers import get_linear_schedule_with_warmup,set_seed
    scheduler=get_linear_schedule_with_warmup(optimizer,10,420); set_seed(42)
    metadata={'status':'running','dataset':name,'condition':method,'seed':42,'protocol_sha256':sha(PROTOCOL),
        'initial_adapter_sha256':initial,'optimizer_steps_completed':0,'optimizer_steps_planned':420,'started_at_kst':trainer.now()}
    write(summary,metadata)
    evaluate(model,dev,tok,run/'epoch_0.json',name+' '+method+' epoch 0')
    original=read(OUT/name/'validation/student.json')
    assert [r['predicted'] for r in read(run/'epoch_0.json')['records']]==[r['predicted'] for r in original['records']]
    torch.cuda.synchronize(); start=perf_counter(); excluded=0; global_step=0
    try:
        with (run/'train_log.jsonl').open('x',encoding='utf-8') as log, (run/'signal_log.jsonl').open('x',encoding='utf-8') as signals:
            for epoch in range(3):
                for group in gate.batches(gate.epoch_order(len(items),42,epoch),8):
                    batch=0.
                    for i in group:
                        signal={}; value=gate.loss(model,examples[i],values[i],method,signal)
                        (value/len(group)).backward(); number=float(value.detach()); batch+=number
                        signal.update(source_id=items[i]['source_id'],epoch=epoch+1,optimizer_step=global_step+1,loss=number)
                        signals.write(json.dumps(signal)+'\n')
                    norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                    lr=optimizer.param_groups[0]['lr']; optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                    global_step+=1; elapsed=perf_counter()-start-excluded
                    record={'epoch':epoch+1,'optimizer_step':global_step,'mean_loss':batch/len(group),
                            'grad_norm':float(norm),'learning_rate':lr,'training_seconds':elapsed,
                            'sample_ids':[items[i]['source_id'] for i in group]}
                    log.write(json.dumps(record)+'\n'); log.flush(); signals.flush()
                    if global_step%8==0:
                        metadata.update(optimizer_steps_completed=global_step,training_seconds=elapsed,last_loss=record['mean_loss'])
                        write(summary,metadata)
                        write(PUBLIC/'status.json',{'stage':'training','dataset':name,'condition':method,'step':global_step,'planned_steps':420})
                        print(f'{name}/{method}: step {global_step}/420, loss={record["mean_loss"]:.4f}',flush=True)
                torch.cuda.synchronize(); elapsed=perf_counter()-start-excluded; begin=perf_counter()
                point=evaluate(model,dev,tok,run/f'epoch_{epoch+1}.json',name+' '+method+f' epoch {epoch+1}')
                write(run/f'epoch_{epoch+1}_timing.json',{'training_seconds':elapsed,'step':global_step})
                torch.cuda.synchronize(); excluded+=perf_counter()-begin
        assert global_step==420 and trainer.adapter_hash(model)!=initial
        assert model.get_input_embeddings().weight.grad is None
        model.save_pretrained(run/'adapter',safe_serialization=True,save_embedding_layers=False)
        (run/'adapter/README.md').unlink(missing_ok=True)
        torch.cuda.synchronize()
        metadata.update(status='passed',optimizer_steps_completed=global_step,training_seconds=perf_counter()-start-excluded,
            adapter_files_sha256={f.name:sha(f) for f in (run/'adapter').iterdir() if f.is_file()},
            final_validation_metrics=point['metrics'],epoch_order_sha256=gate.order_hashes(items,42))
        if method=='adaptive_soft':
            write(run/'final_gradient_audit.json',gradient_audit(model,parameters,items,values,tok))
    except BaseException as exc:
        metadata.update(status='failed',error=str(exc)); raise
    finally:
        metadata['finished_at_kst']=trainer.now(); write(summary,metadata)


def score(name,tag,split):
    require()
    if split=='holdout':
        seal=read(PUBLIC/'evaluation_manifest.json')
        for filename,h in seal['files_sha256'].items(): assert sha(ROOT/filename)==h,filename
    path=OUT/name/split/f'{tag}.json'
    if path.exists():
        assert read(path)['protocol_sha256']==sha(PROTOCOL); return
    adapter=OUT/name/'training'/tag/'adapter' if tag in METHODS else None
    model=load_model('teacher' if tag=='teacher' else 'student',adapter)
    result=evaluate(model,rows(DATA/name/f'{split}.jsonl'),tokenizer(),path,name+' '+tag+' '+split)
    if tag in METHODS and split=='validation':
        previous=read(OUT/name/'training'/tag/'epoch_3.json')
        assert [r['predicted'] for r in result['records']]==[r['predicted'] for r in previous['records']]
        assert np.allclose([r['choice_log_probabilities'] for r in result['records']],
                           [r['choice_log_probabilities'] for r in previous['records']],atol=1e-4,rtol=1e-4)


def freeze_evaluation():
    require(); path=PUBLIC/'evaluation_manifest.json'
    if path.exists():
        for filename,h in read(path)['files_sha256'].items(): assert sha(ROOT/filename)==h,filename
        return
    hashes={str(PROTOCOL.relative_to(ROOT)):sha(PROTOCOL)}
    for name in DATASETS:
        for method in METHODS:
            run=OUT/name/'training'/method; v=read(run/'summary.json')
            assert v['status']=='passed' and v['optimizer_steps_completed']==420
            assert (OUT/name/'validation'/f'{method}.json').exists()
            for f,h in v['adapter_files_sha256'].items():
                target=run/'adapter'/f; assert sha(target)==h
                hashes[str(target.relative_to(ROOT))]=h
            hashes[str((run/'summary.json').relative_to(ROOT))]=sha(run/'summary.json')
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    spec=data.SPECS['openbookqa']
    raw=hf_hub_download(repo_id=spec['id'],repo_type='dataset',revision=spec['revision'],
             filename='main/test-00000-of-00001.parquet',cache_dir=str(CACHE))
    forbidden={r['question_sha256'] for split in ('train','validation') for r in rows(DATA/'openbookqa'/f'{split}.jsonl')}
    test,seen=[],set(); excluded=[]; tok=tokenizer()
    for raw_item in pq.read_table(raw).to_pylist():
        item=data.normalize('openbookqa',raw_item,'test'); key=item['question_sha256']
        if key in forbidden or key in seen:
            excluded.append(item['source_id']); continue
        encode(item,tok); seen.add(key); test.append(item)
    save_rows(DATA/'openbookqa/holdout.jsonl',test)
    for name in DATASETS:
        target=DATA/name/'holdout.jsonl'; hashes[str(target.relative_to(ROOT))]=sha(target)
    write(path,{'status':'frozen_before_any_final_scores','frozen_at_kst':trainer.now(),
        'files_sha256':hashes,'openbookqa_raw_test_sha256':sha(raw),'openbookqa_excluded_input_groups':excluded,
        'holdout_counts':{name:len(rows(DATA/name/'holdout.jsonl')) for name in DATASETS},
        'no_post_holdout_training_or_selection':True,'preservation':require_preservation()})
    print('Six adapters and final evaluation inputs frozen; final scoring may begin.',flush=True)


def pipeline():
    require(); require_preservation()
    def child(action,name=None,method=None,split=None):
        args=[sys.executable,'-B','-X','utf8',str(Path(__file__)),action]
        if name: args+=['--dataset',name]
        if method: args+=['--method',method]
        if split: args+=['--split',split]
        subprocess.run(args,cwd=ROOT,check=True)
    for name in DATASETS:
        child('targets',name); child('score',name,'student','validation'); child('probe',name)
    for name,method in read(PROTOCOL)['training_order']:
        child('train',name,method); child('score',name,method,'validation')
    freeze_evaluation()
    for name in DATASETS:
        for tag in ('teacher','student',*METHODS): child('score',name,tag,'holdout')
    subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'src/report_data_gates.py')],cwd=ROOT,check=True)
    write(PUBLIC/'status.json',{'stage':'completed','training_complete':6,'training_planned':6,
         'final_evaluation_complete':10,'preservation':require_preservation(),'finished_at_kst':trainer.now()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','targets','probe','train','score','freeze','pipeline'])
    parser.add_argument('--dataset',choices=DATASETS)
    parser.add_argument('--method',choices=('teacher','student',*METHODS))
    parser.add_argument('--split',choices=['validation','holdout'],default='validation')
    args=parser.parse_args()
    if args.action in ('targets','probe','train','score') and not args.dataset: parser.error('--dataset required')
    if args.action in ('train','score') and not args.method: parser.error('--method required')
    if args.action=='prepare': prepare()
    elif args.action=='targets': teacher_targets(args.dataset)
    elif args.action=='probe': probe(args.dataset)
    elif args.action=='train': train(args.dataset,args.method)
    elif args.action=='score': score(args.dataset,args.method,args.split)
    elif args.action=='freeze': freeze_evaluation()
    else: pipeline()
