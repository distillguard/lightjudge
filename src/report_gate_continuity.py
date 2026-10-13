"""Render the binary/continuous gate comparison from frozen train-only screening."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'outputs/gate_continuity/.matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bayesian_analysis import ROOT, read, sha

public = ROOT / 'results/gate_continuity'
protocol = read(public/'protocol.json')
assert sha(ROOT/'src/gate_continuity.py') == protocol['code_sha256']
result = read(public/'train_screening.json')
assert result['protocol_sha256'] == sha(public/'protocol.json')
raw = {}
for key in ('teacher','student'):
    path = ROOT/f'outputs/gate_continuity/{key}.json'
    assert sha(path) == result['raw_score_sha256'][str(path.relative_to(ROOT))]
    raw[key] = read(path)['observations']

fig, axes = plt.subplots(1,2, figsize=(10.8,4.2), constrained_layout=True)
gap = np.linspace(-1,1,1001)
axes[0].step(gap, (gap>=0).astype(float), where='post', color='#3578a8', label='Hard: 0 or 1')
axes[0].plot(gap, 1/(1+np.exp(-(gap-.05))), color='#cb783b', label='Soft: sigmoid(gap - 0.05)')
axes[0].axhline(.5, color='gray', ls=':', label='Fixed: 0.5')
axes[0].set(xlabel='Teacher - Student confidence', ylabel='Forward-KL weight',
            title='Binary selection vs continuous mixing', ylim=(-.03,1.03))
axes[0].legend(fontsize=8, loc='upper left')
values = []
for dataset in protocol['selected_train_ids']:
    teacher, student = raw['teacher'][dataset], raw['student'][dataset]
    assert [r['source_id'] for r in teacher] == [r['source_id'] for r in student] == protocol['selected_train_ids'][dataset]
    d = np.array([a['confidence']['full_vocabulary']-b['confidence']['full_vocabulary'] for a,b in zip(teacher,student)])
    w = 1/(1+np.exp(-(d-.05)))
    assert len(w)==128 and np.isfinite(w).all()
    assert np.isclose(w.mean(),result['datasets'][dataset]['full_vocabulary']['soft_weight']['mean'])
    values.append(w)
axes[1].boxplot(values, tick_labels=['ARC','OpenBookQA','HellaSwag'], showfliers=True)
axes[1].axhline(.5,color='gray',ls=':',label='Fixed 0.5')
axes[1].axhline(1/(1+np.exp(.05)),color='#cb783b',ls='--',label='Zero-gap 0.4875')
axes[1].set(ylabel='Soft Forward-KL weight', title='Before adaptation: 128 train items each')
axes[1].legend(fontsize=8,loc='lower left')
fig.savefig(public/'hard_soft_data_screening.png',dpi=170)
plt.close(fig)
print('Verified frozen inputs and rendered binary/continuous gate figure.')
