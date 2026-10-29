"""Run one scratch case through the official trainer, measuring complete updates."""

import argparse
import json
import statistics
import time
from pathlib import Path
import yaml
import torch
from transformers import TrainerCallback
from llamafactory.train.tuner import run_exp

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--warmup', type=int, required=True)
p.add_argument('--profile-stages', action='store_true', help='Measure visual and language forward CUDA time')
p.add_argument(
    '--sparse-loss', action='store_true', help='Project only supervised positions into the vocabulary'
)
a = p.parse_args()
config = yaml.safe_load(a.config.read_text())
if a.sparse_loss:
    from sparse_loss import install

    install()
result_path = a.config.parent / 'result.json'
rows = []
events = []
handles = []
stage_records = []


class Timing(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if a.profile_stages:
            model = kwargs['model']
            for name, module in model.named_modules():
                if not name.endswith(('.visual', '.language_model')):
                    continue
                stage = 'vision' if name.endswith('.visual') else 'language_forward'
                pending = []

                def before(module, inputs, stage=stage, pending=pending):
                    start = torch.cuda.Event(enable_timing=True)
                    start.record()
                    pending.append(start)

                def after(module, inputs, output, stage=stage, pending=pending):
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    pooled = getattr(output, 'pooler_output', None)
                    events.append((stage, pending.pop(), end, getattr(pooled, 'requires_grad', None)))

                handles.extend(
                    [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
                )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.previous = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        now = time.monotonic()
        rows.append({'step': state.global_step, 'seconds': now - self.previous})
        if a.profile_stages:
            stages = {}
            for stage, start, end, needs_grad in events:
                record = stages.setdefault(stage, dict(cuda_s=0, calls=0, output_requires_grad=needs_grad))
                record['cuda_s'] += start.elapsed_time(end) / 1000
                record['calls'] += 1
            stage_records.append(dict(step=state.global_step, stages=stages))
            events.clear()
        self.previous = now
        result_path.with_name('step-times.json').write_text(json.dumps(rows, indent=2) + '\n')


start = time.monotonic()
status = 'passed'
error = None
try:
    run_exp(args=config, callbacks=[Timing()])
except Exception as exc:
    status = (
        'oom' if isinstance(exc, torch.OutOfMemoryError) or 'out of memory' in str(exc).lower() else 'failed'
    )
    error = repr(exc)
finally:
    measured = [row['seconds'] for row in rows if row['step'] > a.warmup]
    report = dict(
        status=status,
        error=error,
        wall_s=time.monotonic() - start,
        warmup_updates=a.warmup,
        measured_updates=len(measured),
        mean_update_s=statistics.mean(measured) if measured else None,
        median_update_s=statistics.median(measured) if measured else None,
        stdev_update_s=statistics.stdev(measured) if len(measured) > 1 else None,
        examples_per_second=32 / statistics.mean(measured) if measured else None,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        device_total_mib=torch.cuda.get_device_properties(0).total_memory / 2**20,
        microbatch=config['per_device_train_batch_size'],
        accumulation=config['gradient_accumulation_steps'],
        workers=config['dataloader_num_workers'],
        step_times=rows,
    )
    if a.profile_stages:
        report['stage_profile'] = stage_records
    report['sparse_loss'] = a.sparse_loss
    result_path.write_text(json.dumps(report, indent=2) + '\n')
if status != 'passed':
    raise SystemExit(1)
