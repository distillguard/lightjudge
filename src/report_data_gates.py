"""Paired one-seed external-data results; no model fitting or checkpoint selection."""
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import data_gate_experiment as experiment
from bayesian_analysis import read, rows, sha, write

ROOT, OUT, PUBLIC = experiment.ROOT, experiment.OUT, experiment.PUBLIC
NAMES={'fixed_mix':'Fixed','adaptive_mix':'Hard','adaptive_soft':'Soft'}


def paired(left,right):
    assert [r['source_id'] for r in left]==[r['source_id'] for r in right]
    assert len({r['question_sha256'] for r in left})==len(left)
    a=np.array([r['correct'] for r in left],dtype=float)
    b=np.array([r['correct'] for r in right],dtype=float)
    rng=np.random.default_rng(42)
    indices=rng.integers(0,len(a),size=(5000,len(a)))
    difference=a-b
    nll=np.array([-r['choice_log_probabilities'][ord(r['gold'])-65] for r in left])-np.array(
        [-r['choice_log_probabilities'][ord(r['gold'])-65] for r in right])
    return {'accuracy_difference_pp':float(100*difference.mean()),
            'accuracy_paired_ci95_pp':(100*np.quantile(difference[indices].mean(axis=1),[.025,.975])).tolist(),
            'choice_nll_difference':float(nll.mean()),'choice_nll_paired_ci95':np.quantile(nll[indices].mean(axis=1),[.025,.975]).tolist(),
            'left_only_correct':int(((a==1)&(b==0)).sum()),'right_only_correct':int(((a==0)&(b==1)).sum()),
            'scope':'conditional on one seed and frozen holdout items; secondary intervals unadjusted'}


def main():
    p=experiment.require(); seal=read(PUBLIC/'evaluation_manifest.json')
    for name,h in seal['files_sha256'].items(): assert sha(ROOT/name)==h,name
    summary={'stage':'completed','seed':42,'protocol_sha256':sha(experiment.PROTOCOL),'datasets':{},
             'preservation':experiment.require_preservation(),'new_training_runs':6,'no_post_holdout_training':True}
    for name in experiment.DATASETS:
        scores={tag:read(OUT/name/'holdout'/f'{tag}.json') for tag in ('teacher','student',*experiment.METHODS)}
        for tag,v in scores.items():
            assert v['protocol_sha256']==sha(experiment.PROTOCOL)
            assert len(v['records'])==seal['holdout_counts'][name]
            assert [r['source_id'] for r in v['records']]==[r['source_id'] for r in scores['teacher']['records']]
        result={'holdout_n':seal['holdout_counts'][name],'scores':{tag:v['metrics'] for tag,v in scores.items()},
                'comparisons':{},'training':{},'score_sha256':{tag:sha(OUT/name/'holdout'/f'{tag}.json') for tag in scores}}
        for left,right in [('adaptive_mix','adaptive_soft'),('adaptive_mix','fixed_mix'),('adaptive_soft','fixed_mix')]:
            result['comparisons'][NAMES[left]+'−'+NAMES[right]]=paired(scores[left]['records'],scores[right]['records'])
        for method in experiment.METHODS:
            run=OUT/name/'training'/method
            result['training'][method]=read(run/'summary.json')
            result['training'][method]['validation_curve']=[read(run/f'epoch_{e}.json')['metrics'] for e in range(4)]
        result['gradient_audit']={'initial':read(OUT/name/'probe.json')['gradient_audit'],
            'final_soft':read(OUT/name/'training/adaptive_soft/final_gradient_audit.json')}
        summary['datasets'][name]=result
    write(PUBLIC/'summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,(name,result) in zip(axes,summary['datasets'].items()):
        for method in experiment.METHODS:
            ax.plot([0,140,280,420],[100*v['accuracy'] for v in result['training'][method]['validation_curve']],marker='o',label=NAMES[method])
        ax.set(title=name,xlabel='Optimizer step',ylabel='Development accuracy (%)')
        ax.legend(); ax.grid(alpha=.2)
    fig.savefig(PUBLIC/'validation_curves.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,name in zip(axes,experiment.DATASETS):
        for method in experiment.METHODS:
            log=rows(OUT/name/'training'/method/'train_log.jsonl')
            assert len(log)==420
            loss=np.array([r['mean_loss'] for r in log])
            ax.plot(np.arange(1,421),loss,alpha=.2)
            ax.plot(np.arange(20,421),np.convolve(loss,np.ones(20)/20,mode='valid'),label=NAMES[method])
        ax.set(title=name,xlabel='Optimizer step',ylabel='Loss (raw + 20-step display mean)'); ax.legend()
    fig.savefig(PUBLIC/'training_loss.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for ax,(name,result) in zip(axes,summary['datasets'].items()):
        for i,(comparison,v) in enumerate(result['comparisons'].items()):
            mean=v['accuracy_difference_pp']; low,high=v['accuracy_paired_ci95_pp']
            ax.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt='o')
        ax.axvline(0,color='gray',ls=':'); ax.set(title=name,xlabel='Paired accuracy difference (pp)',
            yticks=range(3),yticklabels=list(result['comparisons']))
    fig.savefig(PUBLIC/'paired_effects.png',dpi=170); plt.close(fig)
    table='| 데이터 | 평가 n | Teacher / 원본 Student | Fixed / Hard / Soft | Hard−Soft paired 95% CI |\n|---|---|---|---|---|\n'
    for name,v in summary['datasets'].items():
        metric=v['scores']; c=v['comparisons']['Hard−Soft']; low,high=c['accuracy_paired_ci95_pp']
        table+=f"| {name} | {v['holdout_n']} | {100*metric['teacher']['accuracy']:.2f}% / {100*metric['student']['accuracy']:.2f}% | "
        table+=' / '.join(f"{100*metric[m]['accuracy']:.2f}%" for m in experiment.METHODS)
        table+=f" | {c['accuracy_difference_pp']:+.2f}%p [{low:+.2f},{high:+.2f}] |\n"
    for filename in ('README.md','docs/RESEARCH_PLAN.md','docs/RESEARCH_REPORT.md'):
        path=ROOT/filename; text=path.read_text(encoding='utf-8'); prefix='' if filename=='README.md' else '../'
        block='<!-- data-gates:start -->\n## 새 데이터 1-seed 탐색 완료: 2026-10-14\n\n'+table
        block+='\n각 train 1,118개·3 epochs·seed 42에서 독립 6회 학습과 개발용 기준선·최종 평가를 완료했다. 최종 epoch 3·수식·checkpoint는 고정했고 최종 평가 후 재학습하지 않았다. 선택지 문자 argmax 평가이며 공식 HellaSwag completion 점수나 기존 ARC 자유 생성 결과와 직접 비교하지 않는다. CI는 관측한 1 seed에 조건부이고 우위 미확인을 동등성으로 해석하지 않는다.\n\n'
        block+=f'[동결·전체 결과]({prefix}results/data_gates_v1/summary.json) · [개발용 학습곡선]({prefix}results/data_gates_v1/validation_curves.png) · [학습 loss]({prefix}results/data_gates_v1/training_loss.png) · [paired 차이]({prefix}results/data_gates_v1/paired_effects.png)\n'
        block+='<!-- data-gates:end -->\n'
        if '<!-- data-gates:start -->' in text:
            text=re.sub(r'<!-- data-gates:start -->.*?<!-- data-gates:end -->\n*',block,text,flags=re.S)
        else: text+='\n'+block
        text=text.replace('새 데이터 증류 학습 | 미실행','새 데이터 증류 학습 | 완료')
        text=text.replace('**미실행:** 새 데이터 2개×Fixed·Hard·Soft×seed 42의 **6회 탐색 학습**.','**완료:** 새 데이터 2개×Fixed·Hard·Soft×seed 42의 **6회 탐색 학습**. 아래 완료 결과와 동결 명세를 참조한다.')
        path.write_text(text,encoding='utf-8',newline='\n')
    print(json.dumps({name:{'scores':v['scores'],'comparisons':v['comparisons']} for name,v in summary['datasets'].items()}),flush=True)


if __name__=='__main__': main()
