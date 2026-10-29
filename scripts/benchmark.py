"""Screen batch/loader configurations, then repeat the baseline and fastest case."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from contract import LF_REVISION, digest, root_path, write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True, help='Frozen ten-task dataset export')
p.add_argument('--out', type=Path)
a = p.parse_args()
root = root_path()
run = a.run.resolve()
assert (
    subprocess.check_output(
        ['git', '-C', str(root / '.workspace/Show-Harness/third_party/LlamaFactory'), 'rev-parse', 'HEAD'],
        text=True,
    ).strip()
    == LF_REVISION
)
# Import YAML from the trainer environment via this launcher's pipeline command.
import yaml

manifest = json.loads((run / 'dataset-manifest.json').read_text())
for name, h in manifest['export_sha256'].items():
    assert digest(run / name) == h
assert digest(run / 'training-config.yaml') == manifest['training_config_sha256']
base = yaml.safe_load((run / 'training-config.yaml').read_text())
out = (
    a.out
    or root
    / '.workspace'
    / ('libero-object-benchmark-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
).resolve()
out.mkdir(exist_ok=False)
(root / '.workspace/libero-object-benchmark.txt').write_text(str(out) + '\n')
report = dict(
    status='running',
    dataset=str(run),
    source_config_sha256=digest(run / 'training-config.yaml'),
    effective_batch=32,
    cases=[],
    caveat='Small frozen ten-task pilot; production throughput may differ. Same seed and dataset; streamed ordering may differ with worker count. Scratch adapters are never used for evaluation.',
)
env = dict(
    os.environ,
    DISABLE_VERSION_CHECK='1',
    TOKENIZERS_PARALLELISM='false',
    WANDB_DISABLED='true',
    OMP_NUM_THREADS='4',
)
scripts = Path(__file__).parent


def case(name, batch, workers, warmup, measured):
    directory = out / name
    directory.mkdir()
    config = dict(base)
    config.update(
        output_dir=str(directory / 'output'),
        max_steps=warmup + measured,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=32 // batch,
        dataloader_num_workers=workers,
        do_eval=False,
        eval_strategy='no',
        save_strategy='no',
        load_best_model_at_end=False,
        logging_steps=1,
        seed=42,
    )
    for key in ('eval_dataset', 'resume_from_checkpoint'):
        config.pop(key, None)
    path = directory / 'config.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(
        f'Starting {name}: batch {batch}, accumulation {32 // batch}, workers {workers}, {warmup}+{measured} updates',
        flush=True,
    )
    with (directory / 'log.txt').open('w') as log:
        process = subprocess.Popen(
            [
                sys.executable,
                '-u',
                str(scripts / 'benchmark_case.py'),
                '--config',
                str(path),
                '--warmup',
                str(warmup),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            code = process.wait(timeout=1800)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    result = (
        json.loads((directory / 'result.json').read_text())
        if (directory / 'result.json').exists()
        else dict(status='failed', error='See case log')
    )
    result.update(name=name, returncode=code, config_sha256=digest(path))
    if code and result['status'] == 'passed':
        result['status'] = 'failed'
    report['cases'].append(result)
    write(out / 'report.json', report)
    print(json.dumps({k: v for k, v in result.items() if k != 'step_times'}), flush=True)
    # Official trainer saves a final scratch adapter even with save_strategy=no.
    # Keep timing/config/logs; remove unneeded scratch weights after process exit.
    shutil.rmtree(directory / 'output', ignore_errors=True)
    return result


with (root / '.workspace/libero-object-benchmark.lock').open('w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(out / 'report.json', report)
    try:
        for batch in (8, 4, 16):
            case(f'screen-b{batch}-w2', batch, 2, 3, 8)
        passed = [r for r in report['cases'] if r['status'] == 'passed']
        assert passed, 'No configurations passed'
        best = min(passed, key=lambda r: r['mean_update_s'])
        case(f"screen-b{best['microbatch']}-w4", best['microbatch'], 4, 3, 8)
        best = min((r for r in report['cases'] if r['status'] == 'passed'), key=lambda r: r['mean_update_s'])
        case('confirm-baseline', 8, 2, 5, 20)
        case('confirm-candidate', best['microbatch'], best['workers'], 5, 20)
        confirmed = [
            r for r in report['cases'] if r['name'].startswith('confirm-') and r['status'] == 'passed'
        ]
        assert len(confirmed) == 2, 'Confirmation failed'
        baseline = next(r for r in confirmed if r['name'] == 'confirm-baseline')
        candidate = next(r for r in confirmed if r['name'] == 'confirm-candidate')
        # Require >3% gain to avoid changing defaults for minor timing noise.
        winner = candidate if candidate['mean_update_s'] < baseline['mean_update_s'] * 0.97 else baseline
        report.update(
            status='complete',
            recommended=dict(
                microbatch=winner['microbatch'],
                accumulation=winner['accumulation'],
                workers=winner['workers'],
                seconds_per_update=winner['mean_update_s'],
                examples_per_second=winner['examples_per_second'],
                speedup_vs_baseline=baseline['mean_update_s'] / winner['mean_update_s'],
            ),
        )
    except BaseException as exc:
        report.update(
            status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc)
        )
        raise
    finally:
        write(out / 'report.json', report)
print('Benchmark complete:', out, flush=True)
