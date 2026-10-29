"""Side-by-side comparison of two evaluations (e.g. v5 checkpoint 3464 vs the v6 variant).

Reads each evaluation's summary.json and frozen-evaluation.json; optionally the two
check_changepoints.py summaries. Refuses to compare different budgets or attempt counts,
and prints the protocol of each side so a turn-guarded run is never mistaken for a frozen one.
"""

import argparse
import collections
import json
from pathlib import Path
from contract import write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline', type=Path, required=True, help='Evaluation directory (eval-native-...)')
p.add_argument('--candidate', type=Path, required=True)
p.add_argument('--baseline-changepoints', type=Path, help='check_changepoints output directory')
p.add_argument('--candidate-changepoints', type=Path)
p.add_argument('--out', type=Path, help='Optional JSON report')
a = p.parse_args()


def load(directory):
    summary = json.loads((directory / 'summary.json').read_text())
    frozen = json.loads((directory / 'frozen-evaluation.json').read_text())
    assert summary['status'] == 'completed', f'{directory} is not a completed evaluation'
    return summary, frozen


def describe(summary, frozen):
    rows = summary['rollouts']
    per = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        c = per[r['task']]
        c['attempts'] += 1
        c['success'] += r['success']
        c['all_yaw_cw'] += bool(r['tokens']) and all(t == 'RT_YAW_CW' for t in r['tokens'])
        c['grasp'] += 'GRASP' in r['tokens']
        c['release'] += 'RELEASE' in r['tokens']
        c[f"end:{r['reason']}"] += 1
        c['guard_fired'] += bool(r.get('turn_guard_decisions'))
    return dict(
        label=f"step {frozen['step']} ({Path(frozen['run']).name})",
        protocol=frozen.get('protocol', 'frozen'),
        turn_guard_deg=frozen.get('turn_guard_deg'),
        budget=frozen['budget'],
        attempts_per_task=frozen['attempts_per_task'],
        adapter_sha256=frozen['sha256'],
        prompt_sha256=frozen['prompt_sha256'],
        successes=sum(r['success'] for r in rows),
        attempts=len(rows),
        tasks={t: dict(c) for t, c in sorted(per.items())},
        prompts={r['task']: r.get('task_prompt', r['task']) for r in rows},
    )


(bs, bf), (cs, cf) = load(a.baseline), load(a.candidate)
assert bf['budget'] == cf['budget'] and bf['attempts_per_task'] == cf['attempts_per_task'], (
    'Different budget or attempt count'
)
base, cand = describe(bs, bf), describe(cs, cf)
for side, d in (('baseline', base), ('candidate', cand)):
    guard = f", turn guard {d['turn_guard_deg']:g} deg" if d['turn_guard_deg'] else ''
    print(
        f"{side:9}: {d['label']}  protocol={d['protocol']}{guard}  budget={d['budget']}  -> {d['successes']}/{d['attempts']}"
    )
print(
    f"\n{'task':40} {'success':>13} {'all-yaw':>9} {'GRASP':>9} {'RELEASE':>9} {'step-limit':>11} {'DONE-fail':>10}"
)
for t in sorted(base['tasks']):
    b, c = base['tasks'][t], cand['tasks'].get(t, {})
    pair = lambda k: f"{b.get(k, 0):>3} -> {c.get(k, 0):<3}"
    print(
        f"{str(base['prompts'][t])[:40]:40} {pair('success'):>13} {pair('all_yaw_cw'):>9} {pair('grasp'):>9} "
        f"{pair('release'):>9} {pair('end:max_tokens'):>11} {pair('end:done'):>10}"
    )
report = dict(baseline=base, candidate=cand)
if a.baseline_changepoints and a.candidate_changepoints:
    cb = json.loads((a.baseline_changepoints / 'summary.json').read_text())
    cc = json.loads((a.candidate_changepoints / 'summary.json').read_text())
    assert cb['per_group'] == cc['per_group'], 'Changepoint checks used different --per-group'
    print('\nHeld-out changepoints (teacher-forced):  accuracy  /  copy rate (lower is better)')
    print(
        f"  all changes: {cb['changepoint_accuracy']} -> {cc['changepoint_accuracy']}   /   "
        f"{cb['changepoint_copy_rate']} -> {cc['changepoint_copy_rate']}"
    )
    for g in sorted(set(cb['groups']) | set(cc['groups'])):
        x, y = cb['groups'].get(g, {}), cc['groups'].get(g, {})
        print(
            f"  {g:9} acc {x.get('accuracy')} -> {y.get('accuracy')}   copy {x.get('copy_rate')} -> {y.get('copy_rate')}"
        )
    report.update(baseline_changepoints=cb, candidate_changepoints=cc)
if base['protocol'] != cand['protocol']:
    print('\nWARNING: different protocols; this is not a like-for-like comparison.')
if a.out:
    write(a.out, report)
