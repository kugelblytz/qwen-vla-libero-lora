"""Probe held-out expert start, rotation-to-translation and grasp frames without executing actions."""

import argparse
import base64
import collections
import json
from pathlib import Path
import re
import requests
from contract import digest, write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--adapter', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--url', default='http://127.0.0.1:8000/v1')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
manifest = json.loads((a.run / 'dataset-manifest.json').read_text())
models = requests.get(a.url + '/models', timeout=30)
models.raise_for_status()
registered = next(x for x in models.json()['data'] if x['id'] == 'qwen3_5_4b_libero_v5')
assert Path(registered['root']).resolve() == a.adapter.resolve()
rows = json.loads((a.run / 'preparation-report.json').read_text())['episodes']
results = []
for task in [0, 1, 3, 5]:
    filename = f'validation-task{task:02d}.json'
    assert digest(a.run / filename) == manifest['export_sha256'][filename]
    samples = json.loads((a.run / filename).read_text())
    by_image = {s['images'][0]: s for s in samples}
    for demo in manifest['episode_splits'][str(task)]['validation']:
        row = next(x for x in rows if x['task'] == task and x['demo'] == demo)
        directory = Path(row['rollout'])
        actions = [json.loads(s) for s in (directory / 'actions.jsonl').read_text().splitlines()]
        chosen = [
            ('start', 0),
            ('first_translation', next(i for i, x in enumerate(actions) if x['token'].startswith('MV_'))),
            ('pre_grasp', next(i for i, x in enumerate(actions) if x['token'] == 'GRASP')),
        ]
        for phase, index in chosen:
            sample = by_image[str((directory / actions[index]['agentview']).resolve())]
            content = [
                dict(
                    type='image_url',
                    image_url=dict(
                        url='data:image/png;base64,' + base64.b64encode(Path(image).read_bytes()).decode()
                    ),
                )
                for image in sample['images']
            ]
            content.append(dict(type='text', text=sample['instruction'].removeprefix('<image><image>')))
            response = requests.post(
                a.url + '/chat/completions',
                json=dict(
                    model='qwen3_5_4b_libero_v5',
                    messages=[dict(role='user', content=content)],
                    temperature=0,
                    max_tokens=16,
                    chat_template_kwargs=dict(enable_thinking=False, thinking=False),
                ),
                timeout=120,
            )
            response.raise_for_status()
            raw = response.json()['choices'][0]['message']['content'] or ''
            clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.S).strip()
            tokens = re.findall(r'\b(?:MV_[A-Z_]+|RT_[A-Z_]+|GRASP|RELEASE|DONE)\b', clean)
            predicted = tokens[0] if tokens else None
            result = dict(
                task=task,
                demo=demo,
                phase=phase,
                index=index,
                target=sample['output'],
                predicted=predicted,
                raw=raw,
                correct=predicted == sample['output'],
                images=sample['images'],
            )
            results.append(result)
            write(a.out / 'predictions.json', results)
            print(task, demo, phase, sample['output'], '->', predicted, flush=True)
stats = []
for task in [0, 1, 3, 5]:
    for phase in ['start', 'first_translation', 'pre_grasp']:
        group = [x for x in results if x['task'] == task and x['phase'] == phase]
        stats.append(
            dict(
                task=task,
                phase=phase,
                total=len(group),
                correct=sum(x['correct'] for x in group),
                predictions=dict(collections.Counter(x['predicted'] for x in group)),
            )
        )
write(
    a.out / 'summary.json',
    dict(
        adapter=str(a.adapter.resolve()),
        adapter_sha256=digest(a.adapter / 'adapter_model.safetensors'),
        frames=len(results),
        stats=stats,
        caveat='Expert-state teacher-forced diagnostic on held-out episodes, not rollout success or aggregate frame accuracy.',
    ),
)
