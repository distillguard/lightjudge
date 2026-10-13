"""Existing gate audit and train-only external-data screening; no training or new test access."""
import argparse
import gc
import json
import math
import random
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pilot
from bayesian_analysis import sha, read, rows, write
from teacher_quantization import ROOT, CACHE, MODEL_ID, REVISION, load_text_model

PUBLIC = ROOT / 'results/gate_continuity'
DATA = ROOT / 'data/gate_continuity'
PROTOCOL = PUBLIC / 'protocol.json'
SPECS = {
    'openbookqa': {'id': 'allenai/openbookqa', 'revision': '388097ea7776314e93a529163e0fea805b8a6454',
                   'folder': 'main', 'counts': {'train': 4957, 'validation': 500}},
    'hellaswag': {'id': 'Rowan/hellaswag', 'revision': '218ec52e09a7e7462a5400043bb9a69a41d06b76',
                  'folder': 'data', 'counts': {'train': 39905, 'validation': 10042}},
}


def normalize(name, raw, split):
    if name == 'openbookqa':
        item = {'id': 'openbookqa:' + raw['id'], 'question': raw['question_stem'],
                'choices': raw['choices'], 'answerKey': raw['answerKey']}
    elif name == 'hellaswag':
        if raw['label'] not in ('0', '1', '2', '3') or len(raw['endings']) != 4 or not raw['ctx'].strip():
            raise ValueError('Labeled four-ending HellaSwag rows required; hidden test is unsupported')
        item = {'id': 'hellaswag:' + raw['source_id'] + ':' + str(raw['ind']),
                'question': 'Choose the most plausible continuation of this situation:\n' + raw['ctx'],
                'choices': {'label': list('ABCD'), 'text': raw['endings']},
                'answerKey': 'ABCD'[int(raw['label'])]}
    else:
        raise ValueError('Unknown dataset')
    result = pilot.normalize(item, split)
    if len(result['choices']) != 4:
        raise ValueError('Expected four choices')
    if name == 'hellaswag':
        result.update(scene_id=raw['source_id'], activity=raw['activity_label'], split_type=raw['split_type'])
    return result


def describe(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite nonempty observations required')
    return {'n': len(x), 'mean': float(x.mean()), 'sd_population': float(x.std()),
            'min_p05_p25_median_p75_p95_max': np.quantile(x, [0, .05, .25, .5, .75, .95, 1]).tolist()}


def original_audit():
    from choice_kd import require_preservation
    preserved = require_preservation()
    manifest = read(ROOT / 'results/protocols/gate_test_manifest.json')
    train_ids = {r['source_id'] for r in rows(ROOT / 'data/pilot/train_full.jsonl')}
    result, all_records = {}, []
    for seed in (42, 7, 123):
        path = ROOT / f'outputs/gate_comparison_v4/training/seed_{seed}/adaptive_soft/signal_log.jsonl'
        key = str(path.relative_to(ROOT))
        assert sha(path) == manifest['timing_files_sha256'][key]
        records = rows(path)
        assert len(records) == 3354
        for epoch in (1, 2, 3):
            subset = [r for r in records if r['epoch'] == epoch]
            assert len(subset) == 1118 and {r['source_id'] for r in subset} == train_ids
        gap = np.array([r['teacher_confidence'] - r['student_confidence'] for r in records])
        gate = np.array([r['forward_weight'] for r in records])
        assert np.allclose(gate, 1/(1+np.exp(-(gap-.05))), atol=2e-7, rtol=0)
        result[str(seed)] = {'source_sha256': sha(path), 'weight': describe(gate),
            'gap': describe(gap), 'epochs': {str(e): describe([r['forward_weight'] for r in records if r['epoch']==e])
                                           for e in (1, 2, 3)}}
        all_records.extend(records)
    weights = np.array([r['forward_weight'] for r in all_records])
    coefficient = weights*(1-weights)*np.array([r['forward_kl']-r['reverse_kl'] for r in all_records])
    result = {'preservation': preserved, 'per_seed': result, 'pooled_weight': describe(weights),
        'mean_absolute_distance_from_fixed_half': float(np.abs(weights-.5).mean()),
        'mean_absolute_distance_from_zero_gap_constant': float(np.abs(weights-1/(1+math.exp(.05))).mean()),
        'weight_derivative_wrt_gap': describe(weights*(1-weights)),
        'selection_scalar_coefficient_before_common_loss_scale': describe(coefficient),
        'selection_coefficient_scope': 'g*(1-g)*(ForwardKL-ReverseKL); not the parameter gradient norm',
        'zero_gap_weight': 1/(1+math.exp(.05)),
        'theoretical_confidence_gap_bounds': [-1, 1],
        'theoretical_weight_bounds': [1/(1+math.exp(1.05)), 1/(1+math.exp(-.95))],
        'scope': 'Descriptive 3-seed training signals; repeated items/epochs are not independent samples'}
    write(PUBLIC / 'original_gate_audit.json', result)
    return result, all_records


def prepare():
    if PROTOCOL.exists():
        raise FileExistsError('Existing preparation preserved; do not overwrite')
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    audit, signals = original_audit()
    arc = {split: rows(ROOT / f'data/pilot/{"train_full" if split=="train" else split}.jsonl')
           for split in ('train', 'validation', 'test')}
    arc_keys = {split: {r['input_sha256'] for r in items} for split, items in arc.items()}
    info, samples, train_sets = {}, {}, {'arc_challenge': arc['train']}
    data_sha = {'data/pilot/train_full.jsonl': sha(ROOT / 'data/pilot/train_full.jsonl')}
    for name, spec in SPECS.items():
        detail, all_splits = {}, {}
        for split, expected in spec['counts'].items():
            path = hf_hub_download(repo_id=spec['id'], repo_type='dataset', revision=spec['revision'],
                filename=f'{spec["folder"]}/{split}-00000-of-00001.parquet', cache_dir=str(CACHE))
            raw = pq.read_table(path).to_pylist()
            assert len(raw) == expected
            normalized = [normalize(name, r, split) for r in raw]
            assert len({r['source_id'] for r in normalized}) == len(normalized)
            target = DATA / name / f'{split}.jsonl'
            content = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in normalized)
            if target.exists():
                assert target.read_text(encoding='utf-8') == content, 'Changed partial data preserved'
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding='utf-8', newline='\n')
            data_sha[str(target.relative_to(ROOT))] = sha(target)
            keys = {r['input_sha256'] for r in normalized}
            detail[split] = {'n': len(normalized), 'parquet_bytes': Path(path).stat().st_size,
                'parquet_sha256': sha(path), 'normalized_bytes': target.stat().st_size,
                'normalized_sha256': sha(target), 'labels': dict(Counter(r['gold'] for r in normalized)),
                'duplicate_full_inputs': len(normalized)-len(keys),
                'duplicate_questions': len(normalized)-len({r['question_sha256'] for r in normalized}),
                'prompt_characters': describe([len(screen_prompt(r)) for r in normalized]),
                'exact_input_overlap_with_arc': {s: len(keys & k) for s, k in arc_keys.items()}}
            all_splits[split] = normalized
        train_keys = {r['input_sha256'] for r in all_splits['train']}
        val_keys = {r['input_sha256'] for r in all_splits['validation']}
        detail['train_validation_exact_input_overlap'] = len(train_keys & val_keys)
        if name == 'hellaswag':
            detail['train_validation_shared_scene_ids'] = len({r['scene_id'] for r in all_splits['train']} &
                                                               {r['scene_id'] for r in all_splits['validation']})
        answer_texts = {}
        for r in all_splits['train'] + all_splits['validation']:
            correct = r['choices'][ord(r['gold'])-65]['text']
            correct = ' '.join(unicodedata.normalize('NFKC', correct).split())
            answer_texts.setdefault(r['input_sha256'], set()).add(correct)
        conflicts = {key for key, values in answer_texts.items() if len(values)>1}
        detail['conflicting_gold_full_inputs'] = len(conflicts)
        detail['conflicting_gold_source_ids'] = [r['source_id'] for r in all_splits['train']+all_splits['validation']
                                                if r['input_sha256'] in conflicts]
        # Preserve all raw rows; exclude ambiguous labels and held-out overlaps from training candidates.
        eligible = {}
        for r in all_splits['train']:
            if r['input_sha256'] not in val_keys and r['input_sha256'] not in conflicts:
                eligible.setdefault(r['input_sha256'], r)
        train_sets[name] = list(eligible.values())
        detail['unique_train_inputs_excluding_validation_overlap'] = len(eligible)
        info[name] = {**spec, **detail}
        samples[name] = random.Random(42).sample(train_sets[name], 4)
    selection = {name: [r['source_id'] for r in random.Random(42).sample(items, 128)]
                 for name, items in train_sets.items()}
    protocol = {'stage': 'train-only original-model screening; no training or new test evaluation',
        'data_sha256': data_sha, 'selected_train_ids': selection, 'sample_seed': 42, 'n_per_dataset': 128,
        'teacher': {'id': MODEL_ID, 'revision': REVISION, 'quantization': 'NF4 double quant, BF16 compute'},
        'student': {'id': pilot.STUDENT_ID, 'revision': pilot.STUDENT_REVISION, 'dtype': 'BF16, no adapter'},
        'screen_temperature': 2, 'margin': .05, 'max_prefix_tokens': 1024,
        'confidence_spaces': ['full_vocabulary', 'valid_choices_diagnostic_only'],
        'accuracy': 'Train sample valid-letter argmax at T=1; not free-generation or held-out accuracy',
        'common_instruction': 'Select the best answer to this multiple-choice question.',
        'prompt_change': 'Neutral common instruction for all three datasets, including fresh ARC control',
        'selection': 'Uniform random unique train inputs independent of scores; exclude validation overlap and conflicting gold text',
        'dataset_results_not_used_to_select_winner': True, 'code_sha256': sha(Path(__file__)),
        'sources': {'gate': 'https://arxiv.org/html/2609.36734v1',
                    'openbookqa': 'https://aclanthology.org/D18-1260/',
                    'hellaswag': 'https://aclanthology.org/P19-1472/'}}
    write(PUBLIC / 'datasets.json', info)
    write(PUBLIC / 'review_samples.json', samples)
    write(PROTOCOL, protocol)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    w = [r['forward_weight'] for r in signals]
    axes[0].hist(w, bins=55, density=True, color='#3578a8')
    axes[0].axvline(.5, color='black', label='Fixed 0.5')
    axes[0].axvline(audit['zero_gap_weight'], color='#ce7437', ls='--', label='Zero-gap 0.4875')
    axes[0].set(xlabel='Forward weight', ylabel='Density', title='Observed Soft training weights')
    axes[0].legend(fontsize=8)
    gap = np.linspace(-1, 1, 201)
    axes[1].plot(gap, 1/(1+np.exp(-(gap-.05))))
    axes[1].axhline(.5, color='black', ls=':')
    axes[1].set(xlabel='Teacher - Student confidence', ylabel='Forward weight', title='Literal sigmoid, margin 0.05')
    for seed, detail in audit['per_seed'].items():
        axes[2].plot([1,2,3], [detail['epochs'][str(e)]['sd_population'] for e in (1,2,3)], marker='o', label=seed)
    axes[2].set(xlabel='Epoch', ylabel='Weight SD', title='Dispersion narrows during training', xticks=[1,2,3])
    axes[2].legend(title='Seed', fontsize=8)
    fig.savefig(PUBLIC / 'original_gate_distribution.png', dpi=170)
    plt.close(fig)
    print(json.dumps({'prepared': list(info), 'train_validation_parquet_bytes': sum(
        d[s]['parquet_bytes'] for d in info.values() for s in ('train','validation')), 'gate_mean': audit['pooled_weight']['mean']}), flush=True)


def screen_prompt(row):
    return pilot.prompt(row).replace('Select the best answer to this science question.',
                                     'Select the best answer to this multiple-choice question.', 1)


def screen():
    from choice_kd import require_preservation
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    protocol = read(PROTOCOL)
    assert sha(Path(__file__)) == protocol['code_sha256'], 'Frozen screening code changed'
    for name, checksum in protocol['data_sha256'].items():
        assert sha(ROOT / name) == checksum, name
    require_preservation()
    cache = ROOT / 'outputs/gate_continuity'
    cache.mkdir(parents=True, exist_ok=True)
    encoded = {}
    for model_key in ('teacher', 'student'):
        spec = protocol[model_key]
        tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'], cache_dir=str(CACHE),
                                                  local_files_only=True, trust_remote_code=False)
        encoded[model_key] = {}
        for dataset, selected in protocol['selected_train_ids'].items():
            path = ROOT / 'data/pilot/train_full.jsonl' if dataset=='arc_challenge' else DATA / dataset / 'train.jsonl'
            lookup = {r['source_id']: r for r in rows(path)}
            items = []
            for item_id in selected:
                item = lookup[item_id]
                rendered = tokenizer.apply_chat_template([{'role':'user','content':screen_prompt(item)}], tokenize=False,
                                                          add_generation_prompt=True, enable_thinking=False)
                prefix = tokenizer.encode(rendered, add_special_tokens=False)
                labels = [c['label'] for c in item['choices']]
                choice_ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
                assert all(len(ids)==1 and tokenizer.encode(rendered+label, add_special_tokens=False)==prefix+ids
                           for label, ids in zip(labels, choice_ids))
                assert 0 < len(prefix) <= protocol['max_prefix_tokens'], 'Do not truncate or silently resample'
                items.append({'item':item, 'prefix':prefix, 'choice_ids':[ids[0] for ids in choice_ids]})
            encoded[model_key][dataset] = items
    assert [[v['prefix'] for v in items] for items in encoded['teacher'].values()] == [
        [v['prefix'] for v in items] for items in encoded['student'].values()]
    for model_key in ('teacher', 'student'):
        target = cache / f'{model_key}.json'
        if target.exists():
            prior = read(target)
            assert prior['protocol_sha256'] == sha(PROTOCOL)
            print(f'Reusing completed {model_key} screening', flush=True)
            continue
        spec = protocol[model_key]
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16) if model_key=='teacher' else None
        model, _ = load_text_model(spec['id'], spec['revision'], quant)
        model.eval().requires_grad_(False)
        observations = {}
        with torch.inference_mode():
            for dataset, items in encoded[model_key].items():
                results = []
                for i, v in enumerate(items):
                    logits = model(input_ids=torch.tensor([v['prefix']], device='cuda'), use_cache=False,
                                   logits_to_keep=1).logits[0,-1].float()
                    scores = logits[v['choice_ids']]
                    confidence = {}
                    for name, selected_logits in [('full_vocabulary',logits), ('valid_choices_diagnostic_only',scores)]:
                        logp = torch.log_softmax(selected_logits/2, dim=-1)
                        c = 1+(logp.exp()*logp).sum()/math.log(logp.numel())
                        assert torch.isfinite(c) and -.00001 <= c <= 1.00001
                        confidence[name] = float(c)
                    predicted = v['item']['choices'][int(scores.argmax())]['label']
                    results.append({'source_id':v['item']['source_id'], 'confidence':confidence,
                        'correct_choice_argmax':predicted==v['item']['gold'], 'prefix_tokens':len(v['prefix'])})
                    if (i+1)%32==0:
                        print(f'{model_key} / {dataset}: {i+1}/128', flush=True)
                observations[dataset] = results
        write(target, {'protocol_sha256':sha(PROTOCOL), 'observations':observations})
        del model, logits, scores, selected_logits, logp, c
        gc.collect()
        torch.cuda.empty_cache()
    teacher, student = [read(cache/f'{key}.json')['observations'] for key in ('teacher','student')]
    result = {'protocol_sha256':sha(PROTOCOL), 'scope':protocol['stage'], 'datasets':{}}
    for dataset in protocol['selected_train_ids']:
        t, s = teacher[dataset], student[dataset]
        assert [r['source_id'] for r in t] == [r['source_id'] for r in s] == protocol['selected_train_ids'][dataset]
        detail = {'train_sample_n':len(t), 'teacher_train_choice_accuracy':np.mean([r['correct_choice_argmax'] for r in t]).item(),
                  'student_train_choice_accuracy':np.mean([r['correct_choice_argmax'] for r in s]).item(),
                  'prefix_tokens':describe([r['prefix_tokens'] for r in t])}
        for space in protocol['confidence_spaces']:
            gap = np.array([a['confidence'][space]-b['confidence'][space] for a,b in zip(t,s)])
            weight = 1/(1+np.exp(-(gap-.05)))
            detail[space] = {'confidence_gap':describe(gap), 'soft_weight':describe(weight),
                             'hard_forward_fraction':float((gap>=0).mean())}
        result['datasets'][dataset] = detail
    result['raw_score_sha256'] = {str((cache/f'{key}.json').relative_to(ROOT)):sha(cache/f'{key}.json') for key in ('teacher','student')}
    result['preservation'] = require_preservation()
    write(PUBLIC / 'train_screening.json', result)
    print(json.dumps(result['datasets']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','screen'])
    action = parser.parse_args().action
    prepare() if action=='prepare' else screen()
