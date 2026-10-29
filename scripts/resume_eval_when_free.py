"""Wait for GPU memory, then resume the saved native evaluation."""

import os
from pathlib import Path
import subprocess
import sys
import time
from contract import root_path

root = root_path()
run = Path((root / '.workspace/libero-object-1cm5deg-dataset.txt').read_text().strip())
out = Path((root / '.workspace/libero-object-full-eval.txt').read_text().strip())
query = [
    str(root / '.workspace/qwen-venv/bin/python'),
    '-c',
    'import torch; print(torch.cuda.mem_get_info()[0])',
]
while True:
    result = subprocess.run(query, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('GPU memory query failed: ' + result.stderr[-1000:])
    free = int(result.stdout.strip().splitlines()[-1])
    print(f'Waiting for GPU: {free / 2**30:.2f} GiB free; need at least 16 GiB.', flush=True)
    if free >= 16 * 2**30:
        break
    time.sleep(30)
print('GPU available; resuming evaluation.', flush=True)
os.execv(
    sys.executable,
    [
        sys.executable,
        '-u',
        str(Path(__file__).with_name('evaluate.py')),
        '--run',
        str(run),
        '--adapter',
        str(run / 'training-output/checkpoint-3464'),
        '--attempts',
        '10',
        '--budget',
        'native',
        '--resume-evaluation',
        str(out),
    ],
)
