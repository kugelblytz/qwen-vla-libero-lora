"""Estimate sustained update time from the active launch's training log; no GPU work."""

import argparse
import json
from pathlib import Path
from contract import root_path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path)
a = p.parse_args()
root = root_path()
run = (a.run or Path((root / '.workspace/libero-object-1cm5deg-dataset.txt').read_text().strip())).resolve()
launches = sorted((run / 'training-launches').glob('*.json'))
launch = json.loads(launches[-1].read_text()) if launches else None
out = run / ('probe-output' if launch and launch['probe'] else 'training-output')
log = out / 'trainer_log.jsonl'
if not log.exists():
    print('No update timings yet.')
    raise SystemExit(0)
rows = []
for line in log.read_bytes()[launch.get('log_start_bytes', 0) if launch else 0 :].decode().splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue  # A concurrently written trailing line.
    if rows and row.get('current_steps', 0) < rows[-1].get('current_steps', 0):
        rows = []
    if 'loss' in row:
        rows.append(row)
if launch:
    rows = [r for r in rows if r['current_steps'] > launch['completed_updates']]
if not rows:
    print('Waiting for the first logged optimizer update.')
    raise SystemExit(0)
last = rows[-1]
print(f"Updates: {last['current_steps']}/{last['total_steps']}")
# Discard startup and the first ten updates of the current launch.
warmup_end = (launch['completed_updates'] if launch else 0) + 10
steady = [r for r in rows if r['current_steps'] >= warmup_end]


def seconds(value):
    # datetime.timedelta strings may include a day count.
    days = 0
    if 'day' in value:
        prefix, value = value.split(', ')
        days = int(prefix.split()[0])
    h, m, s = map(float, value.split(':'))
    return days * 86400 + h * 3600 + m * 60 + s


if len(steady) < 2:
    print('Collecting timings; sustained estimate needs two log points after the first ten updates.')
else:
    first = steady[0]
    duration = seconds(last['elapsed_time']) - seconds(first['elapsed_time'])
    delta = last['current_steps'] - first['current_steps']
    if duration > 0 and delta > 0:
        rate = duration / delta
        print(f'{rate:.2f} seconds/update; {32 / rate:.2f} training examples/second')
        print(
            f"Remaining compute at this rate: {(last['total_steps'] - last['current_steps']) * rate / 3600:.2f} hours"
        )
        print(
            'Wall time includes any validation/saving between these points; future overhead and speed may vary.'
        )
