"""Three-update official-trainer integration probe on successful diagnostic conversions.

This scratch dataset has no holdout and must never be used as a benchmark model.
It checks multimodal loading, equal-task streaming, optimizer updates and memory.
"""

import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import time
import yaml
from contract import LF_REVISION, ROT_STEP_DEG, STEP_M, digest, root_path, write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out = a.out.resolve()
a.out.mkdir(parents=True, exist_ok=False)
root = root_path()
template = (a.run / 'prompt-v5-1cm-5deg.txt').read_text()
tasks = {}
registry = {}
for result in sorted(a.run.glob('episodes/task-*/demo-*/result.json')):
    row = json.loads(result.read_text())
    if not row['success']:
        continue
    directory = Path(row['rollout'])
    meta = json.loads((directory / 'metadata.json').read_text())
    assert meta['step_m'] == STEP_M and meta['rot_step_deg'] == ROT_STEP_DEG
    samples = []
    history = []
    records = [json.loads(line) for line in (directory / 'actions.jsonl').read_text().splitlines()]
    for record in records + [dict(meta['final_frame'], token='DONE')]:
        samples.append(
            dict(
                instruction='<image><image>'
                + template.format(task=meta['task'], recent_moves=', '.join(history[-5:][::-1]) or 'none'),
                input='',
                output=record['token'],
                images=[str(directory / record[key]) for key in ('agentview', 'wrist')],
            )
        )
        if record['token'].startswith(('MV_', 'RT_')):
            history.append(record['token'])
    tasks.setdefault(row['task'], []).extend(samples)
assert len(tasks) >= 2, 'Need successful diagnostic conversions from at least two tasks'
for task, samples in tasks.items():
    filename = f'probe-task{task:02d}.json'
    write(a.out / filename, samples)
    registry[f'object_probe_{task:02d}'] = dict(
        file_name=filename,
        columns=dict(prompt='instruction', query='input', response='output', images='images'),
    )
write(a.out / 'dataset_info.json', registry)
config = yaml.safe_load((root / 'configs/paper-aligned-4b-20gb.yaml').read_text())
config.pop('num_train_epochs', None)
config.update(
    dataset=','.join(registry),
    dataset_dir=str(a.out),
    model_name_or_path=str(root / '.workspace/models-4b/base'),
    output_dir=str(a.out / 'output'),
    mix_strategy='interleave_over',
    interleave_probs=','.join([str(1 / len(tasks))] * len(tasks)),
    streaming=True,
    accelerator_config={'dispatch_batches': False},
    buffer_size=256,
    max_steps=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    do_eval=False,
    eval_strategy='no',
    save_strategy='no',
    logging_steps=1,
    report_to='none',
    dataloader_num_workers=2,
    seed=42,
)
path = a.out / 'config.yaml'
path.write_text(yaml.safe_dump(config, sort_keys=False))
lf = root / '.workspace/Show-Harness/third_party/LlamaFactory'
assert subprocess.check_output(['git', '-C', str(lf), 'rev-parse', 'HEAD'], text=True).strip() == LF_REVISION
env = dict(os.environ, DISABLE_VERSION_CHECK='1', TOKENIZERS_PARALLELISM='false', WANDB_DISABLED='true')
start = time.monotonic()
peak = 0
with (a.out / 'training.log').open('w') as log:
    process = subprocess.Popen(
        [str(root / '.workspace/lf-paper-venv/bin/llamafactory-cli'), 'train', str(path)],
        cwd=lf,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    while process.poll() is None:
        usage = subprocess.check_output(['nvidia-smi'], text=True)
        numbers = [int(n) for n in re.findall(r'(\d+)MiB\s*/\s*\d+MiB', usage)]
        if numbers:
            peak = max(peak, sum(numbers))
        time.sleep(1)
report = dict(
    status='passed' if process.returncode == 0 else 'failed',
    returncode=process.returncode,
    tasks=sorted(tasks),
    unique_samples=sum(len(rows) for rows in tasks.values()),
    optimizer_updates=3,
    wall_s=time.monotonic() - start,
    peak_device_gpu_mib=peak or None,
    step_m=STEP_M,
    rot_step_deg=ROT_STEP_DEG,
    caveat='Scratch integration probe only; no holdout, no checkpoint selection or rollout-performance claim.',
)
write(a.out / 'report.json', report)
print(json.dumps(report, indent=2))
raise SystemExit(process.returncode)
