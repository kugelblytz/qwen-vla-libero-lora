"""Compare the v6 technique on Qwen3.5-2B (adapter-3500) with the 4B v6 run (adapter-3500).

Usage: analyze_v6_2b.py --v6-4b <summary.json> --v6-2b <eval dir> --out metrics.json
The 4B side needs only its summary.json (tokens per attempt). The 2B eval dir holds summary.json and,
for failed attempts, task-XX-attempt-YYY/rollout/actions.jsonl (pose and gripper width before each token);
failures are classified from the gripper width after the first GRASP, as in results-v6.md section 3.
"""

import argparse, collections, json, statistics as st
from math import comb
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--v6-4b', type=Path, required=True)
ap.add_argument('--v6-2b', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
EMPTY_M = 0.005  # fingers fully closed on nothing (results-v6.md: 0.001 m in every miss)
HELD_M = 0.035  # fingers stopped by the object (results-v6.md: 0.039-0.046 m in every success)


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 2) if xs else None


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def stats(r):
    t = r['tokens']
    g = [i for i, x in enumerate(t) if x == 'GRASP']
    rl = [i for i, x in enumerate(t) if x == 'RELEASE']
    return dict(
        first_grasp=g[0] if g else None,
        n_grasp=len(g),
        n_release=len(rl),
        grasp_to_release=(rl[0] - g[0]) if g and rl and rl[0] > g[0] else None,
        turn_share=sum(x.startswith('RT_') for x in t) / max(1, len(t)),
        all_yaw_cw=bool(t) and all(x == 'RT_YAW_CW' for x in t),
    )


data = {
    '4b': json.load(open(args.v6_4b))['rollouts'],
    '2b': json.load(open(args.v6_2b / 'summary.json'))['rollouts'],
}
for n in data:
    for r in data[n]:
        r.update(stats(r))
key = lambda r: (r['task'], r['initial_state'])
a = {key(r): r for r in data['4b']}
b = {key(r): r for r in data['2b']}
M = {}

print('== 1. Outcomes (paired by task and start)')
both = sum(a[k]['success'] and b[k]['success'] for k in a)
only2 = sum(b[k]['success'] and not a[k]['success'] for k in a)
only4 = sum(a[k]['success'] and not b[k]['success'] for k in a)
M['paired'] = dict(
    both=both,
    only_2b=only2,
    only_4b=only4,
    neither=len(a) - both - only2 - only4,
    mcnemar_p=mcnemar(only2, only4),
)
print(M['paired'])
for n in data:
    M[f'reasons_{n}'] = dict(collections.Counter(r['reason'] for r in data[n]))
    print(n, M[f'reasons_{n}'])

print('\n== 2. Behaviour')
for n in data:
    rs = data[n]
    s = [r for r in rs if r['success']]
    f = [r for r in rs if not r['success']]
    M[f'behaviour_{n}'] = dict(
        all_yaw_cw=sum(r['all_yaw_cw'] for r in rs),
        turn_share=round(
            sum(r['turn_share'] * len(r['tokens']) for r in rs) / sum(len(r['tokens']) for r in rs), 4
        ),
        yaw_over_125=sum(r['max_abs_yaw_deg'] > 125 for r in rs),
        grasped=sum(r['n_grasp'] > 0 for r in rs),
        released=sum(r['n_release'] > 0 for r in rs),
        regrasp_attempts=sum(r['n_grasp'] > 1 for r in rs),
        hit_limit=sum(r['reason'] == 'max_tokens' for r in rs),
        median_first_grasp=med(r['first_grasp'] for r in rs),
        median_grasp_to_release=med(r['grasp_to_release'] for r in rs),
        success_decisions=med(r['num_steps'] for r in s),
        success_control_steps=med(r['control_steps'] for r in s),
        failure_seconds=med(r['wall_s'] for r in f),
        success_seconds=med(r['wall_s'] for r in s),
        successes_with_release=sum(r['n_release'] > 0 for r in s),
        total_hours=round(sum(r['wall_s'] for r in rs) / 3600, 2),
    )
    print(n, M[f'behaviour_{n}'])
same = [k for k in a if a[k]['success'] and b[k]['success']]
M['both_solved_decisions'] = dict(
    n=len(same), v6_4b=med(a[k]['num_steps'] for k in same), v6_2b=med(b[k]['num_steps'] for k in same)
)
print('attempts both solved:', M['both_solved_decisions'])

print('\n== 3. 2B failures by gripper width after the first GRASP')
fails = []
for r in data['2b']:
    if r['success']:
        continue
    d = args.v6_2b / f"task-{r['task']:02d}-attempt-{r['initial_state']:03d}/rollout/actions.jsonl"
    rows = [json.loads(l) for l in d.read_text().splitlines() if l.strip()]
    g = r['first_grasp']
    w_after = rows[g + 1]['gripper_width'] if g is not None and g + 1 < len(rows) else None
    w_end = rows[-1]['gripper_width']
    if g is None:
        grip = 'never grasped'
    elif w_after < EMPTY_M:
        grip = 'closed on nothing'
    elif w_after < HELD_M:
        grip = 'partial grip'
    elif r['n_release']:
        grip = 'held, released outside the basket'
    else:
        grip = 'held, then lost' if w_end < EMPTY_M else 'held, never released'
    P = [row['ee_pose'][:3] for row in rows[g:]] if g is not None else []
    after = collections.Counter(r['tokens'][g + 1 :]) if g is not None else collections.Counter()
    fails.append(
        dict(
            task=r['task'],
            start=r['initial_state'],
            object=r['task_prompt'][9:-27],
            reason=r['reason'],
            grip=grip,
            width_after_grasp=w_after,
            width_end=w_end,
            releases=r['n_release'],
            grasps=r['n_grasp'],
            max_z_after_grasp=round(max(p[2] for p in P), 3) if P else None,
            up_after_grasp=after['MV_UP'],
            right_after_grasp=after['MV_RIGHT'],
        )
    )
M['failures'] = fails
M['failure_table'] = {
    f'{g} / {rs}': c
    for (g, rs), c in sorted(collections.Counter((f['grip'], f['reason']) for f in fails).items())
}
print(M['failure_table'])
M['failures_by_object'] = dict(collections.Counter(f['object'] for f in fails))
print(M['failures_by_object'])

print('\n== 4. Per task')
per = {}
for n in data:
    for r in data[n]:
        p = per.setdefault(r['task'], dict(task=r['task_prompt'][9:-27]))
        p[n] = p.get(n, 0) + r['success']
for t, p in sorted(per.items()):
    print(f"  {p['task']:22} 4B {p['4b']:2}/10  2B {p['2b']:2}/10")
M['per_task'] = per
args.out.write_text(json.dumps(M, indent=1, default=str))
