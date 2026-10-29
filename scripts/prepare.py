"""Prepare the Object suite; explicit --resume continues only matching completed episodes."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from contract import (
    STEP_M,
    ROT_STEP_DEG,
    ROT_TOL_DEG,
    GRIPPER_STEPS,
    MAX_TOKENS,
    SHOW_REVISION,
    digest,
    freeze_prompt,
    root_path,
    simulator_env,
    write,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path)
p.add_argument('--tasks', default='0,1,2,3,4,5,6,7,8,9')
p.add_argument('--episodes-per-task', type=int, default=50)
p.add_argument('--resume', action='store_true')
p.add_argument('--diagnostic', action='store_true', help='Convert only; do not export a training dataset')
a = p.parse_args()
root = root_path()
scripts = Path(__file__).parent
assert 1 <= a.episodes_per_task <= 50
run = (
    a.run
    or root / '.workspace' / ('libero-object-1cm5deg-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
).resolve()
if not a.resume:
    run.mkdir(parents=True, exist_ok=False)
assert run.is_dir()
with (run / 'preparation.lock').open('w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sh = root / '.workspace/Show-Harness'
    assert (
        subprocess.check_output(['git', '-C', str(sh), 'rev-parse', 'HEAD'], text=True).strip()
        == SHOW_REVISION
    )
    prompt = freeze_prompt(root, run)
    settings = dict(
        step_m=STEP_M,
        rot_step_deg=ROT_STEP_DEG,
        rotation_tolerance_deg=ROT_TOL_DEG,
        gripper_steps=GRIPPER_STEPS,
        max_conversion_tokens=MAX_TOKENS,
        prompt_sha256=digest(prompt),
        tasks=a.tasks,
        episodes_per_task=a.episodes_per_task,
        diagnostic=a.diagnostic,
        show_harness_revision=SHOW_REVISION,
        libero_revision=subprocess.check_output(
            ['git', '-C', str(root / '.workspace/LIBERO'), 'rev-parse', 'HEAD'], text=True
        ).strip(),
    )
    if (run / 'settings.json').exists():
        assert json.loads((run / 'settings.json').read_text()) == settings, 'Resume settings mismatch'
    else:
        write(run / 'settings.json', settings)
    env = simulator_env(root)
    sim = root / '.workspace/LIBERO/.venv/bin/python'
    print('Run:', run, flush=True)
    subprocess.run(
        [str(sim), str(scripts / 'list_tasks.py'), '--run', str(run), '--tasks', a.tasks], env=env, check=True
    )
    subprocess.run(
        [
            str(root / '.workspace/qwen-venv/bin/python'),
            str(scripts / 'download_sources.py'),
            '--run',
            str(run),
        ],
        check=True,
    )
    results = []
    for task in json.loads((run / 'tasks.json').read_text()):
        for demo in range(a.episodes_per_task):
            out = run / 'episodes' / f"task-{task['index']:02d}" / f'demo-{demo:03d}'
            if not (out / 'result.json').exists():
                if out.exists():
                    raise RuntimeError(
                        f'Incomplete conversion at {out}; inspect before retrying in a fresh run'
                    )
                print(f"Task {task['index']}, episode {demo}", flush=True)
                log = run / f"conversion-task{task['index']:02d}-demo{demo:03d}.log"
                with log.open('x') as f:
                    subprocess.run(
                        [
                            str(sim),
                            '-u',
                            str(scripts / 'convert_episode.py'),
                            '--run',
                            str(run),
                            '--task',
                            str(task['index']),
                            '--demo',
                            str(demo),
                        ],
                        env=env,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            row = json.loads((out / 'result.json').read_text())
            results.append(row)
            assert row['task'] == task['index'] and row['demo'] == demo
            print(
                f"  {row['status']}: success={row['success']}; control steps={row.get('control_steps')}",
                flush=True,
            )
            write(
                run / 'preparation-report.json',
                dict(
                    episodes=results,
                    successes=sum(r['success'] for r in results),
                    attempted=len(results),
                    status='preparing',
                ),
            )
    write(
        run / 'preparation-report.json',
        dict(
            episodes=results,
            successes=sum(r['success'] for r in results),
            attempted=len(results),
            status='diagnostic_complete' if a.diagnostic else 'converted',
        ),
    )
    if not a.diagnostic:
        subprocess.run(
            [
                str(root / '.workspace/lf-paper-venv/bin/python'),
                str(scripts / 'export_dataset.py'),
                '--run',
                str(run),
            ],
            check=True,
        )
        pointer = root / '.workspace/libero-object-1cm5deg-dataset.txt'
        pointer.write_text(str(run) + '\n')
    print('Preparation complete:', run, flush=True)
