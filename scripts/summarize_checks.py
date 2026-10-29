"""Preserve increment checks, conversion outcomes and two-camera diagnostic videos."""

import argparse
import json
from pathlib import Path
import subprocess
import shutil
import tempfile
from contract import ROT_STEP_DEG, STEP_M, digest, root_path, write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--controls', type=Path, required=True)
p.add_argument('--probe', type=Path, required=True)
a = p.parse_args()
root = root_path()
controls = json.loads((a.controls / 'control-check.json').read_text())
probe = json.loads((a.probe / 'report.json').read_text())
assert controls['status'] == 'passed' and probe['status'] == 'passed'
episodes = []
for file in sorted(a.run.glob('episodes/task-*/demo-*/result.json')):
    row = json.loads(file.read_text())
    episodes.append(row)
    if 'rollout' not in row:
        continue
    rollout = Path(row['rollout'])
    meta = json.loads((rollout / 'metadata.json').read_text())
    assert meta['step_m'] == STEP_M and meta['rot_step_deg'] == ROT_STEP_DEG
    assert meta['follower_rotation_tolerance_deg'] == 3
    assert meta['prompt_sha256'] == digest(a.run / 'prompt-v5-1cm-5deg.txt')
    # The final post-success frame exists only for successful conversions.
    # Encode all available action observations; this is a conversion video,
    # not a trained-policy evaluation or a fresh cumulative replay.
    video = rollout / 'conversion.mp4'
    if (rollout / 'agentview/0000.png').exists() and not video.exists():
        with tempfile.TemporaryDirectory(prefix='object-conversion-video-') as temporary:
            directory = Path(temporary)
            (directory / 'rollout').symlink_to(rollout.resolve(), target_is_directory=True)
            if meta.get('final_frame'):
                (directory / 'final-agentview.png').symlink_to(
                    (rollout / meta['final_frame']['agentview']).resolve()
                )
                (directory / 'final-wrist.png').symlink_to((rollout / meta['final_frame']['wrist']).resolve())
            subprocess.run(
                [
                    str(root / '.workspace/qwen-venv/bin/python'),
                    str(root / 'scripts/sim/make_rollout_video.py'),
                    str(directory),
                ],
                check=True,
            )
            shutil.move(str(directory / 'trajectory.mp4'), str(video))
        row['video'] = str(video)
    elif video.exists():
        row['video'] = str(video)
tasks = sorted({row['task'] for row in episodes if row['success']})
summary = dict(
    status='passed',
    controls=controls,
    training_probe=probe,
    converted_task_coverage=tasks,
    converted_successes=sum(row['success'] for row in episodes),
    candidates=len(episodes),
    episodes=episodes,
    caveat='Demonstration conversion and a short training integration probe, not a learned-policy benchmark. Failed candidates are excluded from training.',
)
write(a.run / 'verification-summary.json', summary)
print('Successful task coverage:', tasks)
print('Diagnostic candidates:', len(episodes), 'successful', summary['converted_successes'])
