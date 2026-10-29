"""Held-out action-changepoint check against a served checkpoint (no simulator).

Validation frames are grouped by the expert's label:
  first     first frame of an episode
  switch    label differs from the previous label (excluding GRASP/RELEASE/DONE)
  GRASP / RELEASE / DONE
  continue  label equals the previous label (control group)
For every non-continue frame the report also gives the copy rate: how often the model
predicted the PREVIOUS expert label instead, i.e. kept doing what it was doing. This is
the failure behind the RT_YAW_CW loops and is what changepoint oversampling targets.
Works for v5 and variant runs alike (it reads the frozen validation exports as-is).
"""

import argparse
import base64
import collections
import json
from pathlib import Path
import re
import requests
from contract import digest, served_adapter_name, write

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--run', type=Path, required=True)
p.add_argument(
    '--adapter', type=Path, required=True, help='Must match the adapter registered at the endpoint'
)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--url', default='http://127.0.0.1:8000/v1')
p.add_argument('--per-group', type=int, default=8, help='Frames per group per task (evenly spread)')
p.add_argument('--tasks', default='0,1,2,3,4,5,6,7,8,9')
a = p.parse_args()
assert a.per_group >= 1
a.out.mkdir(parents=True, exist_ok=False)
manifest = json.loads((a.run / 'dataset-manifest.json').read_text())
models = requests.get(a.url + '/models', timeout=30)
models.raise_for_status()
registered = next(m for m in models.json()['data'] if m['id'] == served_adapter_name(manifest))
assert Path(registered['root']).resolve() == a.adapter.resolve(), 'Endpoint is serving a different checkpoint'


def group_of(label, previous):
    if previous is None:
        return 'first'
    if label in ('GRASP', 'RELEASE', 'DONE'):
        return label
    return 'continue' if label == previous else 'switch'


def spread(indices, n):
    if len(indices) <= n:
        return indices
    return [indices[round(j * (len(indices) - 1) / (n - 1))] for j in range(n)] if n > 1 else indices[:1]


results = []
for t in [int(x) for x in a.tasks.split(',')]:
    filename = f'validation-task{t:02d}.json'
    assert digest(a.run / filename) == manifest['export_sha256'][filename], filename
    samples = json.loads((a.run / filename).read_text())
    previous_label = []
    last_demo = None
    prev = None
    for s in samples:
        demo = re.search(r'/(demo-\d+)/', s['images'][0]).group(1)
        if demo != last_demo:
            prev = None
            last_demo = demo
        previous_label.append(prev)
        prev = s['output']
    groups = collections.defaultdict(list)
    for i, s in enumerate(samples):
        groups[group_of(s['output'], previous_label[i])].append(i)
    for group, indices in sorted(groups.items()):
        for i in spread(indices, a.per_group):
            s = samples[i]
            content = [
                dict(
                    type='image_url',
                    image_url=dict(
                        url='data:image/png;base64,' + base64.b64encode(Path(image).read_bytes()).decode()
                    ),
                )
                for image in s['images']
            ]
            content.append(dict(type='text', text=s['instruction'].removeprefix('<image><image>')))
            response = requests.post(
                a.url + '/chat/completions',
                json=dict(
                    model=served_adapter_name(manifest),
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
            results.append(
                dict(
                    task=t,
                    index=i,
                    group=group,
                    target=s['output'],
                    previous=previous_label[i],
                    predicted=predicted,
                    correct=predicted == s['output'],
                    copied_previous=previous_label[i] is not None
                    and predicted == previous_label[i]
                    and predicted != s['output'],
                )
            )
            print(t, group, s['output'], '->', predicted, flush=True)
            write(a.out / 'predictions.json', results)
summary = {}
for group in sorted({r['group'] for r in results}):
    g = [r for r in results if r['group'] == group]
    summary[group] = dict(
        n=len(g),
        accuracy=round(sum(r['correct'] for r in g) / len(g), 3),
        copy_rate=None if group == 'continue' else round(sum(r['copied_previous'] for r in g) / len(g), 3),
    )
changes = [r for r in results if r['group'] not in ('continue', 'first')]
write(
    a.out / 'summary.json',
    dict(
        run=str(a.run.resolve()),
        variant=manifest.get('variant', 'v5'),
        adapter=str(a.adapter.resolve()),
        adapter_sha256=digest(a.adapter / 'adapter_model.safetensors'),
        per_group=a.per_group,
        groups=summary,
        changepoint_accuracy=round(sum(r['correct'] for r in changes) / len(changes), 3) if changes else None,
        changepoint_copy_rate=round(sum(r['copied_previous'] for r in changes) / len(changes), 3)
        if changes
        else None,
        caveat='Teacher-forced on held-out expert frames (expert history and state). Not rollout success.',
    ),
)
print(json.dumps(summary, indent=1))
