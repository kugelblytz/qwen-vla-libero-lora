"""Serve an explicit frozen checkpoint and evaluate all ten tasks, preserving every video.

--turn-guard-deg runs a DIAGNOSTIC evaluation outside the frozen benchmark contract: its output
directory, frozen-evaluation.json and summary.json are all marked, and its score must be
reported separately from guard-free results. See ../docs/turn-guard.md.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import time
import urllib.request
from contract import ROT_STEP_DEG, STEP_M, digest, prompt_file, root_path, serve_profile, simulator_env, write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--adapter', type=Path, required=True)
p.add_argument('--attempts', type=int, choices=[10, 50], default=10)
p.add_argument('--budget', choices=['matched', 'native'], default='matched')
p.add_argument('--port', type=int, default=8000)
p.add_argument('--dry-run', action='store_true')
p.add_argument(
    '--turn-guard-deg',
    type=float,
    default=0,
    help='DIAGNOSTIC, not the frozen benchmark: withhold yaw past this net angle (e.g. 90); 0 = off. See ../docs/turn-guard.md',
)
p.add_argument(
    '--resume-evaluation',
    type=Path,
    help='Continue an evaluation directory with matching frozen settings; preserve completed videos',
)
a = p.parse_args()
root = root_path()
run = a.run.resolve()
adapter = a.adapter.resolve()
scripts = Path(__file__).parent
manifest = json.loads((run / 'dataset-manifest.json').read_text())
assert manifest['status'] == 'prepared', 'Diagnostic probe adapters cannot be benchmarked'
assert manifest['step_m'] == STEP_M and manifest['rot_step_deg'] == ROT_STEP_DEG
assert a.turn_guard_deg >= 0
assert digest(prompt_file(run, manifest)) == manifest['prompt_sha256']
assert (
    digest(root / '.workspace/Show-Harness/models/chat_templates/qwen3_5_nothink.jinja')
    == manifest['chat_template_sha256']
)
assert digest(root / '.workspace/Show-Harness/configs/robot_libero.yaml') == manifest['robot_config_sha256']
assert adapter.parent == (run / 'training-output').resolve(), 'Select a retained checkpoint from this run'
assert (adapter / 'adapter_model.safetensors').is_file()
state = json.loads((adapter / 'trainer_state.json').read_text())
selection = dict(
    adapter=str(adapter),
    sha256=digest(adapter / 'adapter_model.safetensors'),
    step=state['global_step'],
    run=str(run),
    attempts_per_task=a.attempts,
    budget=a.budget,
    step_m=STEP_M,
    rot_step_deg=ROT_STEP_DEG,
    prompt_sha256=manifest['prompt_sha256'],
)
if a.turn_guard_deg:
    # Only added when set, so guard-free evaluations keep their original frozen-evaluation.json.
    selection.update(
        turn_guard_deg=a.turn_guard_deg,
        protocol='diagnostic-turn-guard',
        protocol_note='Not the frozen benchmark: yaw beyond the guard is withheld. Report separately; see turn-guard.md',
    )
protocol = selection.get('protocol', 'frozen')
env = dict(os.environ, QWEN_PROFILE=serve_profile(manifest), LIBERO_ADAPTER=str(adapter), PORT=str(a.port))
serve = root / 'scripts/sim/serve_qwen.sh'
subprocess.run(['bash', str(serve)], env=dict(env, DRY_RUN='1'), check=True)
if a.dry_run:
    print(json.dumps(selection, indent=2))
    raise SystemExit(0)
with socket.socket() as sock:
    sock.bind(('127.0.0.1', a.port))
if a.resume_evaluation:
    out = a.resume_evaluation.resolve()
    assert out.parent == run, 'Evaluation directory must belong to this dataset run'
    assert json.loads((out / 'frozen-evaluation.json').read_text()) == selection, (
        'Resume evaluation settings mismatch'
    )
else:
    out = run / (
        'eval-'
        + a.budget
        + (f'-turnguard{a.turn_guard_deg:g}' if a.turn_guard_deg else '')
        + '-'
        + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    )
    out.mkdir(exist_ok=False)
    write(out / 'frozen-evaluation.json', selection)
server_log = (out / 'server.log').open('a')
server = None
active = None
active_episode = None


def interrupt(signum, frame):
    raise KeyboardInterrupt()


signal.signal(signal.SIGTERM, interrupt)
try:
    server = subprocess.Popen(['bash', str(serve)], env=env, stdout=server_log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 600
    while True:
        assert server.poll() is None, 'Model server exited; inspect server.log'
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health', timeout=2):
                break
        except Exception:
            if time.monotonic() > deadline:
                raise TimeoutError('Model server startup timed out')
            time.sleep(2)
    rows = []
    for task in range(10):
        for attempt in range(a.attempts):
            episode = out / f'task-{task:02d}-attempt-{attempt:03d}'
            if episode.exists():
                saved = json.loads((episode / 'smoke-check.json').read_text())
                assert saved['status'] == 'completed' and (episode / 'trajectory.mp4').is_file(), (
                    'Incomplete episode: inspect before resuming'
                )
                report = saved['episode']
                rows.append({**report, 'task_prompt': report['task'], 'task': task, 'initial_state': attempt})
                write(
                    out / 'summary.json',
                    dict(
                        protocol=protocol,
                        turn_guard_deg=a.turn_guard_deg or None,
                        attempts=len(rows),
                        successes=sum(r['success'] for r in rows),
                        rollouts=rows,
                        status='running',
                    ),
                )
                print('Preserved', episode.name, flush=True)
                continue
            active_episode = episode
            print(episode.name, flush=True)
            with (out / f'{episode.name}.log').open('w') as f:
                active = subprocess.Popen(
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
                        str(attempt),
                        '--url',
                        f'http://127.0.0.1:{a.port}/v1',
                        '--budget',
                        a.budget,
                        '--turn-guard-deg',
                        str(a.turn_guard_deg),
                    ],
                    env=simulator_env(root),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
                returncode = active.wait()
            video = subprocess.run(
                [
                    str(root / '.workspace/qwen-venv/bin/python'),
                    str(root / 'scripts/sim/make_rollout_video.py'),
                    str(episode),
                ]
            )
            if returncode or video.returncode:
                write(
                    out / 'interruption.json',
                    dict(
                        task=task, attempt=attempt, returncode=returncode, video_returncode=video.returncode
                    ),
                )
                raise RuntimeError('Episode or video failed; partial evidence retained')
            report = json.loads((episode / 'smoke-check.json').read_text())['episode']
            rows.append({**report, 'task_prompt': report['task'], 'task': task, 'initial_state': attempt})
            write(
                out / 'summary.json',
                dict(
                    protocol=protocol,
                    turn_guard_deg=a.turn_guard_deg or None,
                    attempts=len(rows),
                    successes=sum(r['success'] for r in rows),
                    rollouts=rows,
                    status='running',
                ),
            )
            active = None
            active_episode = None
    write(
        out / 'summary.json',
        dict(
            protocol=protocol,
            turn_guard_deg=a.turn_guard_deg or None,
            attempts=len(rows),
            successes=sum(r['success'] for r in rows),
            rollouts=rows,
            status='completed',
        ),
    )
finally:
    if active is not None and active.poll() is None:
        active.send_signal(signal.SIGINT)
        try:
            active.wait(timeout=20)
        except subprocess.TimeoutExpired:
            active.kill()
            active.wait()
    if (
        active_episode is not None
        and active_episode.exists()
        and not (active_episode / 'trajectory.mp4').exists()
    ):
        subprocess.run(
            [
                str(root / '.workspace/qwen-venv/bin/python'),
                str(root / 'scripts/sim/make_rollout_video.py'),
                str(active_episode),
            ]
        )
    if server is not None:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
    server_log.close()
