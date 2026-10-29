"""Offline check of all-yaw rollouts: does the wrist actually turn, and do the cameras change?

Reads rollout/actions.jsonl of completed evaluation attempts. Each record's ee_pose is
logged before its token executes, so the pose change from step i to i+1 is token i's effect.
"""

import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from contract import ROT_STEP_DEG, write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    '--evaluation', type=Path, required=True, help='eval-native-* directory with task-XX-attempt-YYY folders'
)
p.add_argument('--out', type=Path, required=True, help='JSON output, e.g. evidence/yaw-loop-poses.json')
p.add_argument(
    '--quat-order',
    choices=['wxyz', 'xyzw'],
    default='wxyz',
    help='ee_pose[3:7] order; smoke-01 shows gripper-down as ~[0,1,0,0], i.e. wxyz',
)
p.add_argument('--sheets', type=Path, help='Directory for frame contact sheets of looping attempts')
p.add_argument('--sheet-steps', default='0,10,50,200,399')
p.add_argument('--max-sheets', type=int, default=3)
a = p.parse_args()


def rotation(q):
    w, x, y, z = (q[0], q[1], q[2], q[3]) if a.quat_order == 'wxyz' else (q[3], q[0], q[1], q[2])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def heading(R):
    # Finger axis (gripper y) projected onto the table plane; robust to small pitch/roll.
    return math.degrees(math.atan2(R[1, 1], R[0, 1]))


def tilt(R):
    # Angle between gripper approach axis (z) and straight down.
    return math.degrees(math.acos(max(-1.0, min(1.0, -R[2, 2]))))


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def frame_change(root, a_path, b_path):
    x = np.asarray(Image.open(root / a_path).convert('L'), dtype=np.float32)
    y = np.asarray(Image.open(root / b_path).convert('L'), dtype=np.float32)
    return float(np.abs(x - y).mean())


def longest_run(tokens, token):
    best = (0, 0)
    start = None
    for i, t in enumerate(tokens + [None]):
        if t == token and start is None:
            start = i
        elif t != token and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def analyse(episode):
    rollout = episode / 'rollout'
    rows = [json.loads(l) for l in (rollout / 'actions.jsonl').read_text().splitlines() if l.strip()]
    report = json.loads((episode / 'smoke-check.json').read_text())['episode']
    tokens = [r['token'] for r in rows]
    poses = [np.array(r['ee_pose'], dtype=float) for r in rows]
    Rs = [rotation(q[3:7]) for q in poses]
    hs = [heading(R) for R in Rs]
    ts = [tilt(R) for R in Rs]
    yaw_deltas = []
    frame_deltas = []
    for i in range(len(rows) - 1):
        if tokens[i].startswith('RT_YAW'):
            yaw_deltas.append(dict(step=i, token=tokens[i], deg=wrap(hs[i + 1] - hs[i])))
            if all(
                rollout.joinpath(rows[j][v]).is_file() for j in (i, i + 1) for v in ('wrist', 'agentview')
            ):
                frame_deltas.append(
                    dict(
                        step=i,
                        wrist=frame_change(rollout, rows[i]['wrist'], rows[i + 1]['wrist']),
                        agentview=frame_change(rollout, rows[i]['agentview'], rows[i + 1]['agentview']),
                    )
                )
    start, end = longest_run(tokens, 'RT_YAW_CW')
    run = [d for d in yaw_deltas if start <= d['step'] < end]
    executed = lambda ds: (sum(abs(d['deg']) >= ROT_STEP_DEG / 2 for d in ds) / len(ds)) if ds else None
    late = run[len(run) // 2 :]
    unwrapped = np.degrees(np.unwrap(np.radians(hs)))
    drift_mm = float(np.linalg.norm(poses[end - 1][:3] - poses[start][:3]) * 1000) if end > start else 0.0
    tilt_change = float(ts[end - 1] - ts[start]) if end > start else 0.0
    # Pure yaw should keep the TCP in place; the first >10 mm displacement marks the arm breaking away.
    away = (
        np.linalg.norm(np.array([q[:3] for q in poses[start:end]]) - poses[start][:3], axis=1) * 1000
        if end > start
        else np.zeros(0)
    )
    break_step = start + int(np.argmax(away > 10)) if (away > 10).any() else None
    turned_before_break = float(unwrapped[break_step] - unwrapped[start]) if break_step is not None else None
    if len(run) < 20:
        label = 'not-a-yaw-loop'
    elif break_step is not None:
        label = 'breaks-away'
    elif executed(late) < 0.2:
        label = 'plateau'
    elif executed(late) > 0.8:
        label = 'drifting' if drift_mm > 20 or abs(tilt_change) > 10 else 'spinning'
    else:
        label = 'mixed'
    wrist = [f['wrist'] for f in frame_deltas if start <= f['step'] < end]
    return dict(
        episode=episode.name,
        task=report['task'],
        success=report['success'],
        reason=report['reason'],
        num_steps=len(tokens),
        longest_yaw_cw_run=[start, end],
        label=label,
        yaw_executed_fraction_run=executed(run),
        yaw_executed_fraction_second_half=executed(late),
        mean_abs_yaw_step_deg_run=float(np.mean([abs(d['deg']) for d in run])) if run else None,
        signed_mean_yaw_step_deg_run=float(np.mean([d['deg'] for d in run])) if run else None,
        heading_total_unwrapped_deg=float(unwrapped[-1] - unwrapped[0]),
        heading_range_deg=float(unwrapped.max() - unwrapped.min()),
        run_position_drift_mm=drift_mm,
        run_tilt_change_deg=tilt_change,
        break_away_step=break_step,
        yaw_turned_before_break_deg=turned_before_break,
        start_tilt_deg=ts[0],
        mean_wrist_frame_change_run=float(np.mean(wrist)) if wrist else None,
        mean_wrist_frame_change_last20=float(np.mean(wrist[-20:])) if wrist else None,
        first_non_rotation_index=next((i for i, t in enumerate(tokens) if not t.startswith('RT_')), None),
        issued_grasp='GRASP' in tokens,
        yaw_steps=[[d['step'], round(d['deg'], 2)] for d in yaw_deltas],
    )


def contact_sheet(episode, steps, path):
    rollout = episode / 'rollout'
    rows = [json.loads(l) for l in (rollout / 'actions.jsonl').read_text().splitlines() if l.strip()]
    steps = [s for s in steps if s < len(rows)]
    tiles = []
    for s in steps:
        av = Image.open(rollout / rows[s]['agentview']).convert('RGB')
        wr = Image.open(rollout / rows[s]['wrist']).convert('RGB').resize(av.size)
        tile = Image.new('RGB', (av.width, av.height * 2))
        tile.paste(av, (0, 0))
        tile.paste(wr, (0, av.height))
        tiles.append(tile)
    sheet = Image.new('RGB', (sum(t.width for t in tiles), tiles[0].height))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width
    sheet.save(path, quality=85)
    return dict(path=str(path), steps=steps, layout='columns=steps, top=agentview, bottom=wrist')


episodes = sorted(
    e
    for e in a.evaluation.iterdir()
    if e.is_dir() and e.name.startswith('task-') and (e / 'rollout/actions.jsonl').is_file()
)
assert episodes, 'No task-XX-attempt-YYY/rollout/actions.jsonl found'
results = [analyse(e) for e in episodes]
sheets = []
if a.sheets:
    a.sheets.mkdir(parents=True, exist_ok=True)
    steps = [int(s) for s in a.sheet_steps.split(',')]
    loops = [
        r
        for r in results
        if r['label'] != 'not-a-yaw-loop'
        and r['first_non_rotation_index'] is None
        and (a.evaluation / r['episode'] / 'rollout/agentview/0000.png').is_file()
    ]
    for r in loops[: a.max_sheets]:
        sheets.append(contact_sheet(a.evaluation / r['episode'], steps, a.sheets / f"{r['episode']}.jpg"))
counts = {}
for r in results:
    counts[r['label']] = counts.get(r['label'], 0) + 1
write(
    a.out,
    dict(
        evaluation=str(a.evaluation.resolve()),
        quat_order=a.quat_order,
        rot_step_deg=ROT_STEP_DEG,
        label_counts=counts,
        contact_sheets=sheets,
        episodes=results,
    ),
)
print(
    f"{'episode':28} {'label':15} {'ok':3} {'run':>9} {'exec%':>6} {'|step|':>7} {'total':>8} {'drift':>6} {'wrist':>7} {'break@':>6} {'turned':>6}"
)
for r in results:
    if r['label'] == 'not-a-yaw-loop' and not r['success']:
        continue
    f = lambda v, fmt: format(v, fmt) if v is not None else '-'
    print(
        f"{r['episode']:28} {r['label']:15} {'Y' if r['success'] else '':3} "
        f"{r['longest_yaw_cw_run'][1] - r['longest_yaw_cw_run'][0]:>9} {f(r['yaw_executed_fraction_run'] and 100 * r['yaw_executed_fraction_run'], '6.0f')} "
        f"{f(r['mean_abs_yaw_step_deg_run'], '7.2f')} {r['heading_total_unwrapped_deg']:8.0f} {r['run_position_drift_mm']:6.0f} "
        f"{f(r['mean_wrist_frame_change_run'], '7.2f')} {str(r['break_away_step'] if r['break_away_step'] is not None else '-'):>6} "
        f"{f(r['yaw_turned_before_break_deg'], '6.0f')}"
    )
print('labels:', counts)
