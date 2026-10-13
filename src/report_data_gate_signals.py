"""Descriptive gate distributions from completed external-data training logs; no model changes."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import data_gate_experiment as experiment
from bayesian_analysis import read, rows, sha, write
from gate_reporting import describe_signals


def main():
    experiment.require()
    assert read(experiment.PUBLIC/'status.json')['stage']=='completed'
    result={'protocol_sha256':sha(experiment.PROTOCOL),'analysis_source_sha256':sha(Path(__file__)),
        'reused_statistics_source_sha256':sha(Path(__file__).with_name('gate_reporting.py')),
        'scope':'T=2 full-vocabulary training distributions including dropout; descriptive forward-pass variation, not training-seed uncertainty',
        'datasets':{}}
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for ax,name in zip(axes,experiment.DATASETS):
        values={}
        for method in experiment.METHODS:
            path=experiment.OUT/name/'training'/method/'signal_log.jsonl'
            assert read(path.parent/'summary.json')['status']=='passed'
            records=rows(path)
            assert len(records)==3354
            summary=describe_signals(records,method)
            summary['epochs']=[describe_signals([r for r in records if r['epoch']==epoch],method) for epoch in (1,2,3)]
            assert all(v['n_training_forwards']==1118 for v in summary['epochs'])
            weights=np.array([r['forward_weight'] for r in records])
            summary['middle_90_percent']=np.quantile(weights,[.05,.95]).tolist()
            summary['signal_log_sha256']=sha(path)
            values[method]=summary
            if method!='fixed_mix':
                ax.hist(weights,bins=np.linspace(0,1,61),weights=np.ones(len(weights))/len(weights),
                    alpha=.65,color='tab:orange' if method=='adaptive_mix' else 'tab:green',label='Hard (binary)' if method=='adaptive_mix' else 'Soft (continuous)')
        ax.axvline(.5,color='tab:blue',linestyle=':',label='Fixed g=0.5')
        ax.set(title=name,xlabel='Forward KL weight g',ylabel='Fraction of training forward passes',xlim=(0,1))
        ax.legend(fontsize=8)
        result['datasets'][name]=values
    fig.savefig(experiment.PUBLIC/'gate_distributions.png',dpi=170); plt.close(fig)
    write(experiment.PUBLIC/'gate_distributions.json',result)
    table='| 데이터 | Hard Forward 선택 비율 | Soft 평균 / 관측 SD | Soft 가운데 90% 범위 |\n|---|---:|---:|---|\n'
    for name,v in result['datasets'].items():
        hard=v['adaptive_mix']; soft=v['adaptive_soft']; low,high=soft['middle_90_percent']
        table+=f'| {name} | {hard["forward_weight_mean"]:.2%} | {soft["forward_weight_mean"]:.4f} / {soft["forward_weight_sd"]:.4f} | [{low:.4f},{high:.4f}] |\n'
    for filename in ('README.md','docs/RESEARCH_PLAN.md','docs/RESEARCH_REPORT.md'):
        path=experiment.ROOT/filename; text=path.read_text(encoding='utf-8'); prefix='' if filename=='README.md' else '../'
        block='<!-- data-gate-distributions:start -->\n'+table
        block+='\n각 조건의 3,354회 train forward 관측이다. SD는 문항·epoch·dropout에 따른 가중치 변화이며 학습 seed의 불확실성이 아니다. 평균이 0.5에 가까워도 문항별 가중치와 selection gradient는 남아 있으므로 Fixed와 동일하거나 무효하다고 단정하지 않는다.\n\n'
        block+=f'[Gate 분포]({prefix}results/data_gates_v1/gate_distributions.png) · [epoch별 수치·로그 hash]({prefix}results/data_gates_v1/gate_distributions.json)\n<!-- data-gate-distributions:end -->\n'
        assert '<!-- data-gate-distributions:start -->' not in text,'Existing descriptive report preserved'
        text+='\n'+block
        path.write_text(text,encoding='utf-8',newline='\n')
    print(table,flush=True)


if __name__=='__main__': main()
