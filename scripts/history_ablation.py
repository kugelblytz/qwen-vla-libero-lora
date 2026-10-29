"""Diagnostic: rerun fixed starts with the recent-moves history changed, nothing else.

Serve the frozen checkpoint first (RUNNING.md, serve_qwen.sh with LIBERO_ADAPTER). Not a benchmark run.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import urllib.request
from contract import digest, root_path, simulator_env, write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--adapter', type=Path, required=True, help='Checkpoint being served; recorded, not loaded')
p.add_argument(
    '--tasks',
    default='1,5,0',
    help='LIBERO-Object indices; confirm names in the summary (1 cream cheese, 5 tomato sauce, 0 alphabet soup expected)',
)
p.add_argument('--initial-states', default='0,1,2,3,4')
p.add_argument('--variants', default='normal,none,no-rt')
p.add_argument('--budget', choices=['matched', 'native'], default='native')
p.add_argument('--port', type=int, default=8000)
p.add_argument(
    '--reference',
    type=Path,
    help='Full-evaluation summary.json; normal-variant token sequences are compared against it',
)
p.add_argument('--videos', action='store_true')
a = p.parse_args()
root = root_path()
run = a.run.resolve()
adapter = a.adapter.resolve()
scripts = Path(__file__).parent
tasks = [int(t) for t in a.tasks.split(',')]
states = [int(s) for s in a.initial_states.split(',')]
variants = a.variants.split(',')
assert set(variants) <= {'normal', 'none', 'no-rt'}
with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health', timeout=5):
    pass
reference = {}
if a.reference:
    for r in json.loads(a.reference.read_text())['rollouts']:
        reference[(r['task'], r['initial_state'])] = r['tokens']
out = run / ('history-ablation-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
out.mkdir(exist_ok=False)
write(
    out / 'settings.json',
    dict(
        adapter=str(adapter),
        sha256=digest(adapter / 'adapter_model.safetensors'),
        tasks=tasks,
        initial_states=states,
        variants=variants,
        budget=a.budget,
    ),
)


def longest_run(tokens):
    best = cur = 0
    for i, t in enumerate(tokens):
        cur = cur + 1 if i and t == tokens[i - 1] else 1
        best = max(best, cur)
    return best


rows = []
for variant in variants:
    for task in tasks:
        for state in states:
            episode = out / f'{variant}-task-{task:02d}-state-{state:03d}'
            print(episode.name, flush=True)
            with (out / f'{episode.name}.log').open('w') as f:
                code = subprocess.run(
                    [
                        str(root / '.workspace/LIBERO/.venv/bin/python'),
                        '-u',
                        str(scripts / 'eval_episode.py'),
                        '--run',
                        str(run),
                        '--out',
                        str(episode),
                        '--task',
                        str(task),
                        '--initial-state',
                        str(state),
                        '--url',
                        f'http://127.0.0.1:{a.port}/v1',
                        '--budget',
                        a.budget,
                        '--history',
                        variant,
                    ],
                    env=simulator_env(root),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                ).returncode
            if a.videos:
                subprocess.run(
                    [
                        str(root / '.workspace/qwen-venv/bin/python'),
                        str(root / 'scripts/sim/make_rollout_video.py'),
                        str(episode),
                    ]
                )
            if code:
                rows.append(dict(variant=variant, task=task, initial_state=state, error=code))
                continue
            e = json.loads((episode / 'smoke-check.json').read_text())['episode']
            tokens = e['tokens']
            row = dict(
                variant=variant,
                task=task,
                initial_state=state,
                task_prompt=e['task'],
                success=e['success'],
                reason=e['reason'],
                num_steps=len(tokens),
                longest_same_token_run=longest_run(tokens),
                first_non_rotation_index=next(
                    (i for i, t in enumerate(tokens) if not t.startswith('RT_')), None
                ),
                all_yaw_cw=bool(tokens) and all(t == 'RT_YAW_CW' for t in tokens),
                issued_grasp='GRASP' in tokens,
                issued_release='RELEASE' in tokens,
            )
            if variant == 'normal' and (task, state) in reference:
                ref = reference[(task, state)]
                same = next(
                    (i for i, (x, y) in enumerate(zip(tokens, ref)) if x != y), min(len(tokens), len(ref))
                )
                row['matches_reference'] = tokens == ref
                row['reference_agreement_prefix'] = same
            rows.append(row)
            write(out / 'summary.json', dict(status='running', rollouts=rows))
write(out / 'summary.json', dict(status='completed', rollouts=rows))
print(
    f"{'variant':8} {'task':>4} {'state':>5} {'ok':3} {'steps':>5} {'maxrun':>6} {'1st-MV':>6} {'grasp':5} {'ref':4}  prompt"
)
for r in rows:
    if 'error' in r:
        print(r['variant'], r['task'], r['initial_state'], 'ERROR', r['error'])
        continue
    print(
        f"{r['variant']:8} {r['task']:>4} {r['initial_state']:>5} {'Y' if r['success'] else '':3} {r['num_steps']:>5} "
        f"{r['longest_same_token_run']:>6} {str(r['first_non_rotation_index']):>6} {'Y' if r['issued_grasp'] else '':5} "
        f"{str(r.get('matches_reference', '')):4}  {r['task_prompt'][:50]}"
    )
