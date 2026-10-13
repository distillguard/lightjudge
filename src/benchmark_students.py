"""Original-Student ARC validation benchmark, sequenced after the external-data pipeline."""
import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import torch
import choice_kd as kd
import kl_experiment as legacy
import data_gate_experiment as upstream
import monitor_training as diagnostic
import train_student as trainer
from bayesian_analysis import read, sha, write
from teacher_quantization import ROOT, CACHE, load_text_model

PUBLIC = ROOT / 'results/model_benchmark_v1'
OUT = ROOT / 'outputs/model_benchmark_v1'
PROTOCOL = PUBLIC / 'protocol.json'


def require_upstream():
    status = read(upstream.PUBLIC / 'status.json')
    if status.get('stage') != 'completed':
        raise RuntimeError('Finish all six external-data runs and final evaluation before model benchmarking')
    assert status['training_complete'] == status['training_planned'] == 6
    assert status['final_evaluation_complete'] == 10
    upstream.require()
    result = read(upstream.PUBLIC / 'summary.json')
    assert result['stage'] == 'completed' and result['new_training_runs'] == 6
    assert result['protocol_sha256'] == sha(upstream.PROTOCOL)
    for name, expected in read(upstream.PUBLIC / 'evaluation_manifest.json')['files_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for name, values in result['datasets'].items():
        for method in upstream.METHODS:
            summary = values['training'][method]
            assert summary['status'] == 'passed' and summary['optimizer_steps_completed'] == 420
        for tag, checksum in values['score_sha256'].items():
            assert sha(upstream.OUT / name / 'holdout' / (tag + '.json')) == checksum
    return kd.require_preservation()


def freeze():
    if PROTOCOL.exists():
        return require()
    _, _, preparation = kd.require_prepared()
    records, path = legacy.rows_for('full', 'validation')
    assert len(records) == 299 and sha(path) == preparation['validation_sha256']
    prefixes = {}
    for spec in kd.models():
        for name, checksum in spec['tokenizer_file_sha256'].items():
            assert sha(Path(spec['tokenizer_snapshot_path']) / name) == checksum, name
        tok = kd.tokenizer_for(spec)
        examples = [kd.encode_question(r, tok, spec) for r in records]
        hashes = [e['prefix_sha256'] for e in examples]
        assert hashes == preparation['tokenizers'][spec['key']]['prefix_sha256'][1118:]
        prefixes[spec['key']] = hashes
    protocol = {'stage':'frozen_before_benchmark', 'frozen_at_kst':trainer.now(),
        'teacher':'Qwen/Qwen3.5-9B NF4 frozen; Teacher unchanged, this benchmark evaluates original Students only',
        'models':kd.models(), 'source_split':'ARC-Challenge validation', 'validation_n':299,
        'validation_sha256':sha(path), 'validation_source_ids':[r['source_id'] for r in records],
        'prefix_sha256':prefixes, 'precision':'BF16 original base; no LoRA adapter, no training',
        'evaluation':'T=1 valid-choice argmax accuracy and renormalized choice NLL',
        'runtime':'batch 1, use_cache=False, thinking disabled, native templates, one untimed first-prompt warmup',
        'latency_scope':'synchronized single-question prefill, native input token counts differ; shared GPU, not isolated speed benchmark',
        'test_read':False, 'requires_upstream_completed':True,
        'selection_policy':'report all three prespecified Students regardless of accuracy; SmolLM2 parameter-near primary, Granite larger supplementary',
        'source_sha256':{name:sha(ROOT/'src'/name) for name in ('benchmark_students.py','choice_kd.py','pilot.py','teacher_quantization.py','monitor_training.py')},
        'dependencies_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (kd.SETUP, kd.OUT/'preparation.json', ROOT/'results/checks/student_comparison_budget.json', upstream.PROTOCOL)},
        'versions':{k:importlib.metadata.version(k) for k in ('torch','transformers','safetensors')},
        'preservation':kd.require_preservation()}
    write(PROTOCOL, protocol)
    return protocol


def require():
    p = read(PROTOCOL)
    assert p['models'] == kd.models() and p['test_read'] is False
    assert p['versions'] == {k:importlib.metadata.version(k) for k in p['versions']}
    for name, checksum in p['source_sha256'].items():
        assert sha(ROOT/'src'/name) == checksum, name
    for name, checksum in p['dependencies_sha256'].items():
        assert sha(ROOT/name) == checksum, name
    _, path = legacy.rows_for('full','validation')
    assert sha(path) == p['validation_sha256']
    return p


def metrics(records):
    assert records
    losses = [-r['choice_log_probabilities'][r['labels'].index(r['gold'])] for r in records]
    assert np.isfinite(losses).all()
    return {'n':len(records), 'accuracy':float(np.mean([r['predicted']==r['gold'] for r in records])),
            'choice_nll':float(np.mean(losses))}


def load_model(spec):
    from transformers import AutoModelForCausalLM
    if spec['key'] == 'qwen2b':
        model, loading = load_text_model(spec['id'], spec['revision'])
    else:
        model, loading = AutoModelForCausalLM.from_pretrained(spec['id'], revision=spec['revision'],
            cache_dir=str(CACHE),local_files_only=True,trust_remote_code=False,use_safetensors=True,
            dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',output_loading_info=True)
    assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    assert sum(p.numel() for p in model.parameters()) == spec['text_model_parameters']
    assert all(p.dtype == torch.bfloat16 and p.device.type == 'cuda' for p in model.parameters())
    assert not hasattr(model,'peft_config') and not getattr(model,'is_loaded_in_4bit',False)
    model.config.use_cache = False
    return model.eval().requires_grad_(False)


def score(key):
    require_upstream()
    p = require()
    path = OUT / (key + '.json')
    if path.exists():
        old = read(path)
        assert old['status']=='completed' and old['protocol_sha256']==sha(PROTOCOL)
        assert [r['source_id'] for r in old['records']]==p['validation_source_ids']
        assert old['metrics']==metrics(old['records'])
        return
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    spec = next(s for s in p['models'] if s['key']==key)
    tok = kd.tokenizer_for(spec)
    records, _ = legacy.rows_for('full','validation')
    examples = [kd.encode_question(r,tok,spec) for r in records]
    assert [e['prefix_sha256'] for e in examples]==p['prefix_sha256'][key]
    model = load_model(spec)
    result = []
    def forward(example):
        ids = torch.tensor([example['input_ids']],dtype=torch.long,device='cuda')
        return model(input_ids=ids,attention_mask=torch.ones_like(ids),logits_to_keep=1,use_cache=False).logits[0,-1].float()
    with diagnostic.evaluation_scope(model):
        forward(examples[0])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for i,(row,e) in enumerate(zip(records,examples),1):
            torch.cuda.synchronize(); start = perf_counter()
            logits = forward(e)
            torch.cuda.synchronize(); seconds = perf_counter()-start
            assert torch.isfinite(logits).all()
            values = kd.choice_log_probs(logits[e['choice_ids']],temperature=1).cpu().tolist()
            predicted = e['labels'][int(np.argmax(values))]
            result.append({'source_id':row['source_id'],'question_sha256':row['question_sha256'],
                'gold':row['gold'],'labels':e['labels'],'predicted':predicted,'choice_log_probabilities':values,
                'prompt_tokens':e['prompt_tokens'],'prefill_seconds':seconds})
            if i%100==0: print(f'Original {key}: {i}/299',flush=True)
    value = {'status':'completed','model_id':spec['id'],'revision':spec['revision'],
        'protocol_sha256':sha(PROTOCOL),'metrics':metrics(result),'records':result,
        'prompt_token_sum':sum(r['prompt_tokens'] for r in result),
        'mean_prefill_seconds':float(np.mean([r['prefill_seconds'] for r in result])),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'finished_at_kst':trainer.now(),'test_read':False}
    write(path,value)
    print(f'Original {key}: accuracy={value["metrics"]["accuracy"]:.2%}, NLL={value["metrics"]["choice_nll"]:.4f}',flush=True)


def report():
    import re
    p = require()
    budget = read(ROOT/'results/checks/student_comparison_budget.json')
    summary = {'stage':'completed','protocol_sha256':sha(PROTOCOL),'source_split':'validation','test_read':False,
        'models':{},'preservation':require_upstream(),'full_cross_family_training_started':False,
        'selection_policy':p['selection_policy'],'limitations':['Different native token counts, architectures and LoRA counts; no family causal superiority claim','Baseline validation benchmark, not KD outcome','Shared-GPU prefill timing is descriptive only']}
    for spec in p['models']:
        key = spec['key']; path = OUT/(key+'.json'); v = read(path)
        assert v['protocol_sha256']==sha(PROTOCOL) and v['status']=='completed'
        assert [r['source_id'] for r in v['records']]==p['validation_source_ids']
        assert v['metrics']==metrics(v['records'])
        summary['models'][key] = {k:x for k,x in v.items() if k!='records'}
        summary['models'][key].update(score_sha256=sha(path),budget=budget['models'][key])
    write(PUBLIC/'summary.json',summary)
    for filename in ('README.md','docs/RESEARCH_PLAN.md','docs/RESEARCH_REPORT.md'):
        path=ROOT/filename; text=path.read_text(encoding='utf-8'); prefix='' if filename=='README.md' else '../'
        block='<!-- model-benchmark:start -->\n## 다른 계열 원본 Student 벤치마크 완료\n\n'
        block+='Teacher는 Qwen3.5-9B NF4로 유지한다. 새 데이터 6회 학습·최종 평가 완료 뒤, 아래 원본 Student를 공통 ARC validation 299개·T=1 선택지 argmax/NLL로 평가했다. KD 학습 결과가 아니며 모델별 native tokenizer·template를 유지했다.\n\n'
        block+='| 원본 Student | Accuracy | 선택지 NLL |\n|---|---:|---:|\n'
        for v in summary['models'].values():
            block+=f'| {v["model_id"]} | {v["metrics"]["accuracy"]:.2%} | {v["metrics"]["choice_nll"]:.4f} |\n'
        block+=f'\n[벤치마크 결과·토큰 예산]({prefix}results/model_benchmark_v1/summary.json). 독립 선택지 KD 본학습 9회는 아직 미수행이다. 원본 성능만으로 증류 효과나 계열 우위를 결론내리지 않는다.\n<!-- model-benchmark:end -->\n'
        if '<!-- model-benchmark:start -->' in text:
            text=re.sub(r'<!-- model-benchmark:start -->.*?<!-- model-benchmark:end -->\n*',block,text,flags=re.S)
        else: text+='\n'+block
        text=text.replace('원본 성능 벤치마크·독립 본학습 미수행','원본 성능 벤치마크 완료·독립 본학습 미수행')
        text=text.replace('원본 validation 벤치마크·9회 독립 본학습은 미수행이다.','원본 validation 벤치마크는 완료했고 9회 독립 본학습은 미수행이다.')
        text=text.replace('초기 validation·독립 본학습·새 test는 미실행이다.','초기 validation 벤치마크는 완료했고 독립 본학습·새 test는 미실행이다.')
        text=text.replace('새 데이터 증류 학습 | 준비·동결 완료','새 데이터 증류 학습 | 완료')
        text=text.replace('현재 증류 학습·validation 성능 비교·새 test는 미실행이다.','현재 6회 학습·개발용 평가·최종 평가를 완료했다. 아래 완료 결과를 참조한다.')
        path.write_text(text,encoding='utf-8',newline='\n')
    write(PUBLIC/'status.json',{'stage':'completed','students_completed':3,'students_planned':3,'finished_at_kst':trainer.now()})
    print(json.dumps({key:v['metrics'] for key,v in summary['models'].items()}),flush=True)


def pipeline(wait_pid=None):
    require()
    if wait_pid is not None:
        try:
            process=psutil.Process(wait_pid)
            command=process.cmdline()
            assert any('data_gate_experiment.py' in a for a in command) and 'pipeline' in command
            write(PUBLIC/'status.json',{'stage':'waiting_for_external_data','upstream_pid':wait_pid,'students_planned':3})
            print('Waiting for six external-data runs and final scoring; no GPU work starts yet.',flush=True)
            while True:
                try:
                    process.wait(timeout=45)
                    break
                except psutil.TimeoutExpired:
                    pass
        except psutil.NoSuchProcess:
            pass
    require_upstream()
    for spec in kd.models():
        write(PUBLIC/'status.json',{'stage':'benchmarking','student':spec['key'],'students_planned':3})
        subprocess.run([sys.executable,'-B','-X','utf8',str(Path(__file__)),'score','--student',spec['key']],cwd=ROOT,check=True)
    report()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','score','pipeline'))
    parser.add_argument('--student',choices=('qwen2b','smollm2','granite2b'))
    parser.add_argument('--wait-pid',type=int)
    args=parser.parse_args()
    try:
        if args.action=='freeze': freeze()
        elif args.action=='score':
            if not args.student: parser.error('--student required')
            score(args.student)
        else: pipeline(args.wait_pid)
    except Exception as exc:
        if args.action!='freeze': write(PUBLIC/'status.json',{'stage':'failed','error':f'{type(exc).__name__}: {exc}'})
        raise
