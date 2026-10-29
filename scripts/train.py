"""Validate and launch the pinned official trainer; --probe runs only three updates.

Crash-safe use: --resume latest (newest complete checkpoint, or a fresh start), normally via
train_resilient.sh, which relaunches until the configured updates are reached.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import yaml
from contract import LF_REVISION, ROT_STEP_DEG, STEP_M, digest, prompt_file, root_path, write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path)
p.add_argument('--dry-run', action='store_true')
p.add_argument('--probe', action='store_true')
p.add_argument(
    '--resume',
    help='Explicit checkpoint from this run, including optimizer/scheduler state; '
    '"latest" = newest complete checkpoint, or a fresh start if none exists (crash-safe restarts)',
)
p.add_argument(
    '--passes',
    type=int,
    default=5,
    help='Total nominal passes, including already completed updates (default: 5)',
)
p.add_argument(
    '--microbatch',
    type=int,
    choices=(1, 2, 4, 8, 16, 32),
    help='Override microbatch; accumulation adjusts to keep effective batch 32',
)
p.add_argument(
    '--save-steps',
    type=int,
    default=500,
    help='Full checkpoint (adapter+optimizer+scheduler+RNG) every N updates',
)
p.add_argument(
    '--eval-every-passes',
    type=int,
    default=1,
    help='Validation loss every N nominal passes (~15 min each on '
    'the full validation split). Checkpoint selection does not use it; see RUNNING.md',
)
p.add_argument(
    '--keep-checkpoints', type=int, help='Full checkpoints retained (default: max(5, passes)); ~1.6 GB each'
)
p.add_argument(
    '--snapshot-every',
    type=int,
    default=0,
    help='Also keep adapter-only snapshots (training-output/adapter-N, '
    'hard links, ~0.5 GB each) of checkpoints at multiples of N and of the final update, so rotation never deletes '
    'them. 0 = off',
)
a = p.parse_args()
root = root_path()
assert a.passes > 0, 'Passes must be positive'
assert 1 <= a.eval_every_passes <= a.passes, '--eval-every-passes must be between 1 and --passes'
assert a.save_steps > 0 and (a.keep_checkpoints is None or a.keep_checkpoints >= 2) and a.snapshot_every >= 0
assert not a.snapshot_every or a.snapshot_every % a.save_steps == 0, (
    '--snapshot-every must be a multiple of --save-steps'
)
kernel_profile = {}
for package in ('torch', 'triton', 'fla-core', 'flash-linear-attention', 'causal-conv1d'):
    try:
        kernel_profile[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        kernel_profile[package] = None
if os.environ.get('REQUIRE_FAST_KERNELS') == '1':
    from transformers.models.qwen3_5 import modeling_qwen3_5

    assert modeling_qwen3_5.chunk_gated_delta_rule is not None, (
        'Run setup-fast first; optimized Gated DeltaNet unavailable'
    )
    kernel_profile['complete_fast_path'] = modeling_qwen3_5.is_fast_path_available
    print(
        'Optimized Gated DeltaNet enabled; complete fast path:',
        kernel_profile['complete_fast_path'],
        flush=True,
    )
run = (a.run or Path((root / '.workspace/libero-object-1cm5deg-dataset.txt').read_text().strip())).resolve()
lf = root / '.workspace/Show-Harness/third_party/LlamaFactory'
assert subprocess.check_output(['git', '-C', str(lf), 'rev-parse', 'HEAD'], text=True).strip() == LF_REVISION
manifest = json.loads((run / 'dataset-manifest.json').read_text())
assert manifest['status'] == 'prepared' or a.probe or a.dry_run, (
    'Diagnostic exports permit only probes and dry runs'
)
assert manifest['step_m'] == STEP_M and manifest['rot_step_deg'] == ROT_STEP_DEG
assert digest(prompt_file(run, manifest)) == manifest['prompt_sha256']
for name, h in manifest['export_sha256'].items():
    assert digest(run / name) == h, name
assert digest(run / 'dataset_info.json') == manifest['registry_sha256']
assert digest(run / 'training-config.yaml') == manifest['training_config_sha256']
assert (
    digest(root / '.workspace/Show-Harness/models/chat_templates/qwen3_5_nothink.jinja')
    == manifest['chat_template_sha256']
)
assert digest(root / '.workspace/Show-Harness/configs/robot_libero.yaml') == manifest['robot_config_sha256']
config = yaml.safe_load((run / 'training-config.yaml').read_text())
assert (
    config['bf16']
    and config['lora_rank'] == 64
    and config['freeze_vision_tower']
    and config['freeze_multi_modal_projector']
)
assert config['per_device_train_batch_size'] * config['gradient_accumulation_steps'] == 32
assert config['max_steps'] == manifest['max_steps'] and not config['load_best_model_at_end']
steps_per_pass = manifest['updates_per_nominal_epoch']
if a.microbatch:
    config.update(per_device_train_batch_size=a.microbatch, gradient_accumulation_steps=32 // a.microbatch)
config.update(
    max_steps=steps_per_pass * a.passes,
    save_strategy='steps',
    save_steps=a.save_steps,
    eval_steps=steps_per_pass * a.eval_every_passes,
    save_total_limit=a.keep_checkpoints or max(5, a.passes),
    logging_steps=5,
)
if a.probe:
    config.update(
        max_steps=3,
        do_eval=False,
        eval_strategy='no',
        save_strategy='steps',
        save_steps=1,
        save_total_limit=3,
        logging_steps=1,
        output_dir=str(run / 'probe-output'),
        report_to='none',
    )
    config.pop('eval_dataset', None)
out = Path(config['output_dir'])


def checkpoint_problem(path):
    """None if ``path`` is a complete, readable resume point; otherwise the reason it is not.

    A crash during saving can leave a checkpoint directory with missing or truncated files.
    Both torch .pt files are zip archives (truncation breaks the central directory), and a
    safetensors file must be exactly as long as its header declares.
    """
    import zipfile

    for name in (
        'adapter_model.safetensors',
        'adapter_config.json',
        'trainer_state.json',
        'optimizer.pt',
        'scheduler.pt',
    ):
        if not (path / name).is_file() or (path / name).stat().st_size == 0:
            return f'missing {name}'
    if not list(path.glob('rng_state*.pth')):
        return 'missing random-number state'
    try:
        if json.loads((path / 'trainer_state.json').read_text())['global_step'] != int(
            path.name.split('-')[1]
        ):
            return 'trainer_state step does not match directory name'
        for name in ['optimizer.pt', 'scheduler.pt'] + [q.name for q in path.glob('rng_state*.pth')]:
            with zipfile.ZipFile(path / name) as z:
                z.namelist()
        with (path / 'adapter_model.safetensors').open('rb') as f:
            n = int.from_bytes(f.read(8), 'little')
            header = json.loads(f.read(n))
        end = max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
        if (path / 'adapter_model.safetensors').stat().st_size < 8 + n + end:
            return 'adapter_model.safetensors truncated'
    except Exception as e:
        return f'unreadable ({type(e).__name__}: {e})'
    return None


def checkpoints_newest_first():
    found = (
        [q for q in out.glob('checkpoint-*') if q.is_dir() and q.name.split('-')[1].isdigit()]
        if out.exists()
        else []
    )
    return sorted(found, key=lambda q: int(q.name.split('-')[1]), reverse=True)


completed = 0
checkpoint = None
if a.resume == 'latest':
    for candidate in checkpoints_newest_first():
        problem = checkpoint_problem(candidate)
        if problem is None:
            checkpoint = candidate
            break
        print(f'Skipping {candidate.name}: {problem}', flush=True)
    if checkpoint is None and out.exists() and any(out.iterdir()) and not a.dry_run:
        # Crashed before the first complete checkpoint: keep the partial output for inspection, start fresh.
        aside = out.with_name(out.name + '-crashed-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
        out.rename(aside)
        print(f'No complete checkpoint; moved partial output to {aside} and starting fresh', flush=True)
elif a.resume:
    checkpoint = Path(a.resume).resolve()
    assert checkpoint.parent == out.resolve(), 'Resume checkpoint must belong to this run/output'
    problem = checkpoint_problem(checkpoint)
    assert problem is None, f'Cannot resume from {checkpoint}: {problem}'
if checkpoint is not None:
    checkpoint = checkpoint.resolve()
    completed = json.loads((checkpoint / 'trainer_state.json').read_text())['global_step']
    if a.resume == 'latest' and completed >= config['max_steps']:
        print(f'Already complete: {checkpoint.name} reached {config["max_steps"]} updates.', flush=True)
        raise SystemExit(0)
    assert 0 < completed < config['max_steps'], 'Resume target must exceed completed updates'
    config['resume_from_checkpoint'] = str(checkpoint)
elif out.exists() and any(out.iterdir()) and not (a.resume == 'latest' and a.dry_run):
    raise RuntimeError(
        f'Refusing to overwrite {out}; provide an explicit --resume checkpoint or --resume latest'
    )


def snapshot_adapters():
    """Hard-link adapter files of complete checkpoints at multiples of --snapshot-every (and the final
    update) into training-output/adapter-N. checkpoint-* rotation does not touch adapter-* directories,
    and evaluate.py accepts them (same parent, adapter weights + trainer_state.json)."""
    if not a.snapshot_every:
        return
    for q in checkpoints_newest_first():
        step = int(q.name.split('-')[1])
        target = out / f'adapter-{step}'
        if (
            target.exists()
            or (step % a.snapshot_every and step != config['max_steps'])
            or checkpoint_problem(q)
        ):
            continue
        tmp = out / f'.adapter-{step}.tmp'
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        for name in ('adapter_model.safetensors', 'adapter_config.json', 'trainer_state.json'):
            try:
                os.link(q / name, tmp / name)
            except OSError:
                shutil.copy2(q / name, tmp / name)
        tmp.rename(target)
        print(f'Snapshot {target.name}', flush=True)


env = dict(os.environ, DISABLE_VERSION_CHECK='1', TOKENIZERS_PARALLELISM='false', WANDB_DISABLED='true')
env.setdefault('OMP_NUM_THREADS', '4')
print(
    f'Run: {run}\nTotal updates: {config["max_steps"]}; remaining: {config["max_steps"] - completed}\nOutput: {out}',
    flush=True,
)
if checkpoint:
    print(
        'Resume restores adapter, optimizer, scheduler and random-number state. Streaming data replay is not guaranteed bit-identical.',
        flush=True,
    )
    old_total = json.loads((checkpoint / 'trainer_state.json').read_text()).get('max_steps')
    if old_total != config['max_steps']:
        print(
            'Extended budget: cosine/warmup schedule is recalculated for the new total; this differs from an uninterrupted longer run.',
            flush=True,
        )
if a.dry_run:
    os.environ['DISABLE_VERSION_CHECK'] = '1'
    from llamafactory.hparams import get_train_args

    get_train_args(config)
    print('Official configuration parser passed; no training started.')
else:
    with (run / 'training.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another training process holds this run\'s lock; not starting a second one.', flush=True)
            raise SystemExit(75)  # train_resilient.sh stops instead of retrying
        launches = run / 'training-launches'
        launches.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + f'-{os.getpid()}'
        config_path = launches / f'{stamp}.yaml'
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        write(
            launches / f'{stamp}.json',
            dict(
                config_sha256=digest(config_path),
                base_config_sha256=manifest['training_config_sha256'],
                probe=a.probe,
                nominal_passes=None if a.probe else a.passes,
                total_updates=config['max_steps'],
                microbatch=config['per_device_train_batch_size'],
                accumulation=config['gradient_accumulation_steps'],
                loader_workers=config['dataloader_num_workers'],
                omp_num_threads=env['OMP_NUM_THREADS'],
                save_steps=config['save_steps'],
                eval_steps=config.get('eval_steps'),
                save_total_limit=config['save_total_limit'],
                snapshot_every=a.snapshot_every or None,
                kernel_profile=kernel_profile,
                resume_checkpoint=str(checkpoint) if checkpoint else None,
                completed_updates=completed,
                log_start_bytes=(out / 'trainer_log.jsonl').stat().st_size
                if (out / 'trainer_log.jsonl').exists()
                else 0,
                resume_adapter_sha256=digest(checkpoint / 'adapter_model.safetensors')
                if checkpoint
                else None,
            ),
        )
        start = time.monotonic()
        # Use this interpreter so train-fast's isolated compiler/kernel packages
        # are actually used, rather than a CLI with the original venv shebang.
        trainer = subprocess.Popen(
            [sys.executable, '-m', 'llamafactory.cli', 'train', str(config_path)], cwd=lf, env=env
        )

        def interrupt(signum, frame):
            raise KeyboardInterrupt()

        signal.signal(signal.SIGTERM, interrupt)  # stop the trainer too, so no orphan keeps the GPU
        try:
            while trainer.poll() is None:
                time.sleep(30)
                try:
                    snapshot_adapters()
                except Exception as e:
                    print(f'Snapshot check failed ({type(e).__name__}: {e}); training continues', flush=True)
        except BaseException:
            trainer.terminate()
            trainer.wait()
            raise
        snapshot_adapters()
        result = subprocess.CompletedProcess(trainer.args, trainer.returncode)
        elapsed = time.monotonic() - start
        if a.probe:
            write(
                run / 'probe-report.json',
                dict(
                    returncode=result.returncode,
                    wall_s=elapsed,
                    updates=3 - completed,
                    resumed=bool(checkpoint),
                    estimated_full_compute_s=None,
                    caveat='Startup and tokenization included; inspect timed optimizer steps for a better estimate. No production weights selected.',
                ),
            )
        raise SystemExit(result.returncode)
