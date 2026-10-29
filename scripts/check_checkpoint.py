"""Action-stratified free-generation checks against a served candidate checkpoint."""

import argparse
import base64
import collections
import json
from pathlib import Path
import re
import requests
from contract import digest, prompt_file, served_adapter_name, write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
p.add_argument(
    '--adapter', type=Path, required=True, help='Must match the adapter registered at the endpoint'
)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--url', default='http://127.0.0.1:8000/v1')
p.add_argument('--per-action', type=int, default=3)
a = p.parse_args()
assert a.per_action >= 1
a.out.mkdir(parents=True, exist_ok=False)
manifest = json.loads((a.run / 'dataset-manifest.json').read_text())
assert digest(prompt_file(a.run, manifest)) == manifest['prompt_sha256']
models = requests.get(a.url + '/models', timeout=30)
models.raise_for_status()
registered = next(m for m in models.json()['data'] if m['id'] == served_adapter_name(manifest))
assert Path(registered['root']).resolve() == a.adapter.resolve(), 'Endpoint is serving a different checkpoint'
results = []
stats = {}
for filename, expected in manifest['export_sha256'].items():
    if not filename.startswith('validation-'):
        continue
    assert digest(a.run / filename) == expected
    samples = json.loads((a.run / filename).read_text())
    groups = collections.defaultdict(list)
    for i, s in enumerate(samples):
        groups[s['output']].append(i)
    for action, indices in groups.items():
        n = min(a.per_action, len(indices))
        selected = {indices[round(j * (len(indices) - 1) / max(1, n - 1))] for j in range(n)}
        for i in sorted(selected):
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
                    dataset=filename,
                    index=i,
                    target=action,
                    predicted=predicted,
                    raw=raw,
                    correct=predicted == action,
                )
            )
            print(filename, i, action, '->', predicted, flush=True)
            write(a.out / 'predictions.json', results)
for action in sorted({r['target'] for r in results}):
    group = [r for r in results if r['target'] == action]
    stats[action] = dict(correct=sum(r['correct'] for r in group), total=len(group))
write(
    a.out / 'summary.json',
    dict(
        adapter=str(a.adapter.resolve()),
        adapter_sha256=digest(a.adapter / 'adapter_model.safetensors'),
        frames=len(results),
        correct=sum(r['correct'] for r in results),
        per_action=stats,
        caveat='Action-stratified diagnostic, not aggregate frame accuracy or rollout success. Use reserved-start validation rollouts as well.',
    ),
)
