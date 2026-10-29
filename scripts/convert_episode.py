"""Replay one raw expert, then regenerate it with the frozen smaller action contract."""

import argparse
import json
from pathlib import Path
import sys
from contract import (
    STEP_M,
    ROT_STEP_DEG,
    ROT_TOL_DEG,
    GRIPPER_STEPS,
    MAX_TOKENS,
    ControlCounter,
    digest,
    import_sim,
    root_path,
    write,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--task', type=int, required=True)
p.add_argument('--demo', type=int, required=True)
a = p.parse_args()
root = root_path()
import_sim(root)
import h5py
import numpy as np
from scripts.trajectory.real2sim.backends.libero import LiberoBackend
from scripts.trajectory.real2sim.atomic_tokenizer import RolloutWriter
from libero_boundary_rendering import install_boundary_rendering

task = next(t for t in json.loads((a.run / 'tasks.json').read_text()) if t['index'] == a.task)
source = next(t for t in json.loads((a.run / 'sources.json').read_text())['files'] if t['task'] == a.task)
out = a.run / 'episodes' / f'task-{a.task:02d}' / f'demo-{a.demo:03d}'
out.mkdir(parents=True, exist_ok=False)
with h5py.File(source['path']) as h:
    names = sorted(h['data'], key=lambda n: int(n.split('_')[-1]))
    assert 0 <= a.demo < len(names)
    raw = h['data'][names[a.demo]]
    start = np.asarray(raw['states'][0])
    actions = np.asarray(raw['actions'])

# Verify original source success with its actual smooth actions. Camera computation
# is deferred because source selection uses only the environment success predicate.
backend = LiberoBackend.make(task['bddl'], settle_steps=0)
install_boundary_rendering(backend)
backend.reset_to_state(start)
counter = ControlCounter(backend)
raw_success = False
for action in actions:
    backend._obs, *_ = backend.env.step(action)
    raw_success = raw_success or backend.success()
raw_report = dict(success=raw_success, steps=counter.count)
backend.close()
if not raw_success:
    write(
        out / 'result.json',
        dict(task=a.task, demo=a.demo, status='source_failed', raw=raw_report, success=False),
    )
    raise SystemExit(0)

current = {}
original_make = LiberoBackend.make


def make(cls, *args, **kw):
    b = original_make(*args, **kw)
    install_boundary_rendering(b)
    current['backend'] = b
    current['counter'] = ControlCounter(b)
    reset = b.reset_to_state

    def reset_count(state):
        reset(state)
        current['counter'].reset()

    b.reset_to_state = reset_count
    return b


LiberoBackend.make = classmethod(make)
original_close = RolloutWriter.close


def close(writer, meta):
    original_close(
        writer,
        dict(
            meta,
            control_steps=current['counter'].count,
            follower_rotation_tolerance_deg=ROT_TOL_DEG,
            prompt_sha256=digest(a.run / 'prompt-v5-1cm-5deg.txt'),
        ),
    )


RolloutWriter.close = close

follower = root / '.workspace/Show-Harness/scripts/trajectory/real2sim/libero/follow_tokenize.py'
code = follower.read_text()
needle = 'tol_rad = float(np.radians(6.0))'
assert code.count(needle) == 1, 'Pinned follower changed; review local tolerance override'
code = code.replace(needle, f'tol_rad = float(np.radians({ROT_TOL_DEG}))')
# A local, recorded adaptation; the upstream checkout and its revision remain intact.
(out / 'follower-adaptation.json').write_text(
    json.dumps(
        dict(
            upstream_sha256=digest(follower),
            replacement={needle: f'tol_rad = float(np.radians({ROT_TOL_DEG}))'},
        ),
        indent=2,
    )
    + '\n'
)
sys.argv = [
    str(follower),
    '--demos',
    source['path'],
    '--bddl',
    task['bddl'],
    '--out',
    str(out / 'converted'),
    '--demo-ids',
    str(a.demo),
    '--mode',
    'simplified',
    '--fallback-full',
    '--keep-failures',
    '--step-m',
    str(STEP_M),
    '--rot-step-deg',
    str(ROT_STEP_DEG),
    '--gripper-steps',
    str(GRIPPER_STEPS),
    '--max-tokens',
    str(MAX_TOKENS),
]
namespace = dict(__name__='__main__', __file__=str(follower))
try:
    exec(compile(code, str(follower), 'exec'), namespace)
except SystemExit as e:
    if e.code not in (0, 2):
        raise
metadata = next((out / 'converted').glob('*/rollout_*/metadata.json'))
meta = json.loads(metadata.read_text())
write(
    out / 'result.json',
    dict(
        task=a.task,
        demo=a.demo,
        status='converted',
        success=meta['success'],
        reason=meta['reason'],
        rollout=str(metadata.parent),
        raw=raw_report,
        control_steps=meta['control_steps'],
        within_openvla_budget=meta['control_steps'] <= 280,
        actions=sum(1 for _ in (metadata.parent / 'actions.jsonl').open()),
        source_sha256=source['sha256'],
        step_m=meta['step_m'],
        rot_step_deg=meta['rot_step_deg'],
    ),
)
