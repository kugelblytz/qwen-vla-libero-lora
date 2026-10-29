"""Six short episodes: three held-out pre-grasp states, expert and policy control."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request
from contract import digest, root_path, simulator_env, write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--adapter', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--port', type=int, default=8000)
a = p.parse_args()
root = root_path()
scripts = Path(__file__).parent
manifest = json.loads((a.run / 'dataset-manifest.json').read_text())
assert manifest['status'] == 'prepared'
assert a.adapter.resolve().parent == (a.run / 'training-output').resolve()
assert (a.adapter / 'adapter_model.safetensors').is_file()
with socket.socket() as sock:
    sock.bind(('127.0.0.1', a.port))
a.out.mkdir(parents=True, exist_ok=False)
write(
    a.out / 'selection.json',
    dict(
        run=str(a.run.resolve()),
        adapter=str(a.adapter.resolve()),
        adapter_sha256=digest(a.adapter / 'adapter_model.safetensors'),
        tasks=[0, 1, 8],
        max_decisions=24,
        diagnostic='held-out pre-grasp states; expert and model paired controls',
    ),
)
rows = json.loads((a.run / 'preparation-report.json').read_text())['episodes']
env = dict(os.environ, QWEN_PROFILE='4b-libero-v5', LIBERO_ADAPTER=str(a.adapter.resolve()), PORT=str(a.port))
server = None
active = None
results = []


def interrupt(signum, frame):
    raise KeyboardInterrupt()


signal.signal(signal.SIGTERM, interrupt)
try:
    with (a.out / 'server.log').open('w') as log:
        server = subprocess.Popen(
            ['bash', str(root / 'scripts/sim/serve_qwen.sh')], env=env, stdout=log, stderr=subprocess.STDOUT
        )
    deadline = time.monotonic() + 600
    while True:
        assert server.poll() is None, 'Server exited; inspect server.log'
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health', timeout=2):
                break
        except Exception:
            if time.monotonic() > deadline:
                raise TimeoutError('Server startup')
            time.sleep(2)
    for task in [0, 1, 8]:
        demo = manifest['episode_splits'][str(task)]['validation'][0]
        row = next(r for r in rows if r['task'] == task and r['demo'] == demo)
        case = a.out / f'case-task{task:02d}.json'
        write(case, dict(task=task, demo=demo, run=str(a.run.resolve()), rollout=row['rollout']))
        for mode in ['expert', 'model']:
            out = a.out / f'task-{task:02d}-{mode}'
            print('Starting', out.name, flush=True)
            with (a.out / f'{out.name}.log').open('w') as log:
                active = subprocess.Popen(
                    [
                        str(root / '.workspace/LIBERO/.venv/bin/python'),
                        '-u',
                        str(scripts / 'grasp_episode.py'),
                        '--case',
                        str(case),
                        '--mode',
                        mode,
                        '--out',
                        str(out),
                        '--url',
                        f'http://127.0.0.1:{a.port}/v1',
                    ],
                    env=simulator_env(root),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                rc = active.wait()
                active = None
            if out.exists():
                subprocess.run(
                    [
                        str(root / '.workspace/qwen-venv/bin/python'),
                        str(root / 'scripts/sim/make_rollout_video.py'),
                        str(out),
                    ],
                    check=True,
                )
            assert rc == 0, f'Episode failed: {out.name}; inspect log'
            result = json.loads((out / 'result.json').read_text())
            results.append(result)
            print(out.name, result['reason'], flush=True)
            write(a.out / 'summary.json', dict(status='running', episodes=results))
    write(a.out / 'summary.json', dict(status='completed', episodes=results))
finally:
    if active is not None and active.poll() is None:
        active.send_signal(signal.SIGINT)
        try:
            active.wait(timeout=20)
        except subprocess.TimeoutExpired:
            active.kill()
            active.wait()
    if server is not None:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
