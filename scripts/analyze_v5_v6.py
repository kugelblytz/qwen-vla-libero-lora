"""Deep comparison of the v5 (checkpoint 3464) and v6 (adapter-3500) LIBERO-Object evaluations.

Usage: analyze_v5_v6.py --v5 <eval dir> --v6 <eval dir> --out metrics.json
Each eval dir holds summary.json and task-XX-attempt-YYY/rollout/actions.jsonl (pose before each token);
v6 attempts also need decisions.json. Prints sections and writes the metrics JSON.
"""

import argparse, collections, json, math, statistics as st
from math import comb
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--v5', type=Path, required=True)
ap.add_argument('--v6', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
EVALS = {'v5': args.v5, 'v6': args.v6}
OUT = args.out
REVERSE = {
    'MV_LEFT': 'MV_RIGHT',
    'MV_RIGHT': 'MV_LEFT',
    'MV_FWD': 'MV_BACK',
    'MV_BACK': 'MV_FWD',
    'MV_UP': 'MV_DOWN',
    'MV_DOWN': 'MV_UP',
}


def rot(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def yaw_since_start(p0, p):
    r = rot(p[3:7]) @ rot(p0[3:7]).T
    ang = math.acos(max(-1, min(1, (np.trace(r) - 1) / 2)))
    if ang < 1e-9:
        return 0.0
    ax = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if np.linalg.norm(ax) < 1e-6:
        return math.degrees(ang)
    return math.degrees((ax / np.linalg.norm(ax) * ang)[2])


def load(name):
    base = EVALS[name]
    s = json.load(open(base / 'summary.json'))
    out = []
    for r in s['rollouts']:
        d = base / f"task-{r['task']:02d}-attempt-{r['initial_state']:03d}"
        rows = [json.loads(l) for l in (d / 'rollout/actions.jsonl').read_text().splitlines() if l.strip()]
        dec = json.loads((d / 'decisions.json').read_text()) if (d / 'decisions.json').exists() else None
        out.append(dict(r, steps=rows, decisions=dec))
    return out


def phases(r):
    t = r['tokens']
    idx = lambda tok: [i for i, x in enumerate(t) if x == tok]
    g, rl = idx('GRASP'), idx('RELEASE')
    rev = sum(1 for a, b in zip(t, t[1:]) if REVERSE.get(a) == b)
    turns = sum(x.startswith('RT_') for x in t)
    P = [np.array(s['ee_pose'], float) for s in r['steps']]
    stuck = 0
    mv = 0
    for i in range(len(P) - 1):
        if t[i].startswith('MV_'):
            mv += 1
            stuck += np.linalg.norm(P[i + 1][:3] - P[i][:3]) < 0.0025
    yaws = [abs(yaw_since_start(P[0], p)) for p in P] if P else [0]
    return dict(
        first_grasp=g[0] if g else None,
        n_grasp=len(g),
        n_release=len(rl),
        grasp_to_release=(rl[0] - g[0]) if g and rl and rl[0] > g[0] else None,
        release_to_end=(len(t) - rl[-1]) if rl else None,
        reversals_per_100=100 * rev / max(1, len(t)),
        turn_share=turns / max(1, len(t)),
        stuck_share=stuck / max(1, mv),
        max_yaw=max(yaws),
        z_end=float(P[-1][2]) if P else None,
    )


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 2) if xs else None


data = {n: load(n) for n in EVALS}
for n in data:
    for r in data[n]:
        r.update(phases(r))
M = {}

print('== 1. Outcomes (paired by task and start)')
key = lambda r: (r['task'], r['initial_state'])
v5 = {key(r): r for r in data['v5']}
v6 = {key(r): r for r in data['v6']}
both = sum(v5[k]['success'] and v6[k]['success'] for k in v5)
only6 = sum(v6[k]['success'] and not v5[k]['success'] for k in v5)
only5 = sum(v5[k]['success'] and not v6[k]['success'] for k in v5)
neither = 100 - both - only6 - only5
M['paired'] = dict(both=both, only_v6=only6, only_v5=only5, neither=neither, mcnemar_p=mcnemar(only5, only6))
print(M['paired'])
for n in data:
    print(n, 'end reasons', dict(collections.Counter(r['reason'] for r in data[n])))

print('\n== 2. Efficiency')
for n in data:
    s = [r for r in data[n] if r['success']]
    f = [r for r in data[n] if not r['success']]
    M[f'eff_{n}'] = dict(
        success_decisions=med(r['num_steps'] for r in s),
        success_control_steps=med(r['control_steps'] for r in s),
        success_seconds=med(r['wall_s'] for r in s),
        failure_seconds=med(r['wall_s'] for r in f),
        total_hours=round(sum(r['wall_s'] for r in data[n]) / 3600, 2),
    )
    print(n, M[f'eff_{n}'])
same = [k for k in v5 if v5[k]['success'] and v6[k]['success']]
print(
    f'attempts both solved ({len(same)}): decisions v5 {med(v5[k]["num_steps"] for k in same)} vs v6 {med(v6[k]["num_steps"] for k in same)}'
)

print('\n== 3. Action mix and rotation')
for n in data:
    toks = collections.Counter(t for r in data[n] for t in r['tokens'])
    tot = sum(toks.values())
    grp = collections.Counter(
        ('turn' if t.startswith('RT_') else 'move' if t.startswith('MV_') else t) for t in toks.elements()
    )
    my = [r['max_yaw'] for r in data[n]]
    M[f'mix_{n}'] = dict(
        share={g: round(c / tot, 4) for g, c in grp.items()},
        median_max_yaw=med(my),
        yaw_over_90=sum(y > 90 for y in my),
        yaw_over_125=sum(y > 125 for y in my),
        all_yaw_cw=sum(bool(r['tokens']) and all(t == 'RT_YAW_CW' for t in r['tokens']) for r in data[n]),
    )
    print(n, M[f'mix_{n}'])

print('\n== 4. Phases')
for n in data:
    rs = data[n]
    M[f'phase_{n}'] = dict(
        grasped=sum(r['n_grasp'] > 0 for r in rs),
        released=sum(r['n_release'] > 0 for r in rs),
        regrasp_attempts=sum(r['n_grasp'] > 1 for r in rs),
        median_first_grasp=med(r['first_grasp'] for r in rs),
        median_grasp_to_release=med(r['grasp_to_release'] for r in rs),
        successes_with_release=sum(r['success'] and r['n_release'] > 0 for r in rs),
        successes_total=sum(r['success'] for r in rs),
    )
    print(n, M[f'phase_{n}'])

print('\n== 5. Early DONE (v6)')
early = [r for r in data['v6'] if r['reason'] == 'done']
cat = collections.Counter()
for r in early:
    last = r['decisions'][-1].get('robot_state') or ''
    c = (
        'released, then DONE'
        if r['n_release']
        else 'grasped, never released'
        if r['n_grasp']
        else 'never grasped'
    )
    cat[c] += 1
    r['done_gripper'] = 'closed' if 'closed' in last else 'open'
M['early_done'] = dict(
    count=len(early),
    by_history=dict(cat),
    gripper_at_done=dict(collections.Counter(r['done_gripper'] for r in early)),
    median_decisions=med(r['num_steps'] for r in early),
    by_task=dict(collections.Counter(r['task_prompt'][9:30] for r in early)),
)
print(M['early_done'])
for r in early[:30]:
    print(
        f"  task {r['task']} start {r['initial_state']}: {r['num_steps']} decisions, grasps {r['n_grasp']}, releases {r['n_release']}, "
        f"gripper at DONE {r['done_gripper']}, release->DONE {r['release_to_end']}, final z {r['z_end']:.3f}"
    )

print('\n== 6. Back-and-forth and stuck moves')
for n in data:
    rs = data[n]
    M[f'motion_{n}'] = dict(
        reversals_per_100=med(r['reversals_per_100'] for r in rs),
        stuck_move_share=med(r['stuck_share'] for r in rs),
        failures_reversals_per_100=med(r['reversals_per_100'] for r in rs if not r['success']),
        successes_reversals_per_100=med(r['reversals_per_100'] for r in rs if r['success']),
    )
    print(n, M[f'motion_{n}'])
blocked = sum(1 for r in data['v6'] for d in r['decisions'] if d.get('blocked'))
print('v6 decisions with a blocked (stuck-guard) token:', blocked)
M['v6_blocked_decisions'] = blocked

print('\n== 7. Tomato sauce (v6)')
tom = [r for r in data['v6'] if r['task'] == 5]
for r in tom:
    t = r['tokens']
    g = r['first_grasp']
    P = np.array([s['ee_pose'][:3] for s in r['steps']])
    after = t[g + 1 :] if g is not None else []
    right = [i for i in range(g + 1 if g is not None else 0, len(t) - 1) if t[i] == 'MV_RIGHT']
    moved = [np.linalg.norm(P[i + 1] - P[i]) for i in right]
    print(
        f"  start {r['initial_state']}: grasp at {g}, after grasp {len(after)} decisions, MV_RIGHT {len(right)} "
        f"(median move {1000 * st.median(moved) if moved else 0:.1f} mm), y range {P[:, 1].min():.2f}..{P[:, 1].max():.2f}, "
        f"z end {P[-1, 2]:.3f}, max yaw {r['max_yaw']:.0f}, releases {r['n_release']}"
    )
M['tomato'] = [
    dict(
        start=r['initial_state'],
        first_grasp=r['first_grasp'],
        max_yaw=round(r['max_yaw'], 1),
        stuck_share=round(r['stuck_share'], 3),
        releases=r['n_release'],
    )
    for r in tom
]

print('\n== 8. Per task')
per = {}
for n in data:
    for r in data[n]:
        p = per.setdefault(r['task'], dict(task=r['task_prompt'][9:30]))
        p[f'{n}_success'] = p.get(f'{n}_success', 0) + r['success']
        p.setdefault(f'{n}_decisions', []).append(r['num_steps'])
for t, p in sorted(per.items()):
    print(
        f"  {p['task']:22} v5 {p['v5_success']:2}/10  v6 {p['v6_success']:2}/10  median decisions v5 {med(p['v5_decisions'])} v6 {med(p['v6_decisions'])}"
    )
M['per_task'] = {t: {k: v for k, v in p.items() if not k.endswith('decisions')} for t, p in per.items()}
OUT.write_text(json.dumps(M, indent=1, default=str))
