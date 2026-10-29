"""Export successful multi-task trajectories with an episode-level holdout and equal task mixing."""

import argparse
import collections
import json
import math
from pathlib import Path
import random
import yaml
from contract import STEP_M, ROT_STEP_DEG, GRIPPER_STEPS, LF_REVISION, digest, root_path, write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
p.add_argument(
    '--diagnostic-probe',
    action='store_true',
    help='Export diagnostic episodes only for a short integration probe',
)
a = p.parse_args()
root = root_path()
run = a.run.resolve()
assert not (run / 'training-config.yaml').exists(), 'Already exported; refusing to overwrite'
settings = json.loads((run / 'settings.json').read_text())
assert not settings['diagnostic'] or a.diagnostic_probe, 'Diagnostic conversions are not training-ready'
tasks = json.loads((run / 'tasks.json').read_text())
assert len(tasks) == 10, 'Training requires all ten Object tasks'
prompt_path = run / 'prompt-v5-1cm-5deg.txt'
assert digest(prompt_path) == settings['prompt_sha256']
template = prompt_path.read_text()
successes = collections.defaultdict(list)
report = run / ('extended-preparation-report.json' if a.diagnostic_probe else 'preparation-report.json')
for row in json.loads(report.read_text())['episodes']:
    if row['success']:
        successes[row['task']].append(row)
exports = {'train': {}, 'validation': {}}
splits = {}
counts = {}
allowed = {
    'MV_FWD',
    'MV_BACK',
    'MV_LEFT',
    'MV_RIGHT',
    'MV_UP',
    'MV_DOWN',
    'RT_ROLL_LEFT',
    'RT_ROLL_RIGHT',
    'RT_PITCH_FWD',
    'RT_PITCH_BACK',
    'RT_YAW_CW',
    'RT_YAW_CCW',
    'GRASP',
    'RELEASE',
    'DONE',
}
for task in tasks:
    episodes = sorted(successes[task['index']], key=lambda row: row['demo'])
    assert len(episodes) >= 2, (
        f"Task {task['index']}: need at least two successful episodes for an episode-level split"
    )
    random.Random(42 + task['index']).shuffle(episodes)
    n_val = max(1, round(len(episodes) * 0.1))
    splits[str(task['index'])] = {}
    counts[str(task['index'])] = {}
    for split, subset in [('train', episodes[n_val:]), ('validation', episodes[:n_val])]:
        samples = []
        splits[str(task['index'])][split] = [row['demo'] for row in subset]
        for row in subset:
            directory = Path(row['rollout'])
            meta = json.loads((directory / 'metadata.json').read_text())
            assert meta['success'] and meta['step_m'] == STEP_M and meta['rot_step_deg'] == ROT_STEP_DEG
            assert (
                meta['gripper_steps'] == GRIPPER_STEPS and meta['prompt_sha256'] == settings['prompt_sha256']
            )
            assert meta['task_key'] == task['name']
            for field, value in dict(
                agentview_rotation_degrees=180,
                agentview_flip='none',
                wrist_rotation_degrees=0,
                wrist_flip='none',
            ).items():
                assert meta[field] == value, (field, meta[field])
            records = [json.loads(line) for line in (directory / 'actions.jsonl').read_text().splitlines()]
            import numpy as np

            with np.load(directory / 'states.npz') as states:
                assert len(states['states']) == len(records) + 1
            assert meta['final_frame']['state_index'] == len(records)
            history = []
            for record in records + [dict(meta['final_frame'], token='DONE')]:
                token = record['token']
                assert token in allowed
                images = [str((directory / record[key]).resolve()) for key in ('agentview', 'wrist')]
                assert all(Path(image).is_file() for image in images)
                samples.append(
                    dict(
                        instruction='<image><image>'
                        + template.format(
                            task=meta['task'], recent_moves=', '.join(history[-5:][::-1]) or 'none'
                        ),
                        input='',
                        output=token,
                        images=images,
                    )
                )
                if token.startswith(('MV_', 'RT_')):
                    history.append(token)
        exports[split][task['index']] = samples
        counts[str(task['index'])][split] = len(samples)
        assert samples
registry = {}
hashes = {}
names = {}
for split in ('train', 'validation'):
    names[split] = []
    for task, samples in exports[split].items():
        filename = f'{split}-task{task:02d}.json'
        write(run / filename, samples)
        hashes[filename] = digest(run / filename)
        name = f'object_1cm5deg_{split}_{task:02d}'
        names[split].append(name)
        registry[name] = dict(
            file_name=filename,
            columns=dict(prompt='instruction', query='input', response='output', images='images'),
        )
write(run / 'dataset_info.json', registry)
config = yaml.safe_load((root / 'configs/paper-aligned-4b-20gb.yaml').read_text())
total = sum(counts[str(t)]['train'] for t in exports['train'])
# HF interleave over ten datasets uses equal probabilities and an explicit epoch
# length. This avoids larger tasks dominating while retaining all trajectories.
steps_per_epoch = math.ceil(total / 32)
config.update(
    model_name_or_path=str(root / '.workspace/models-4b/base'),
    dataset=','.join(names['train']),
    eval_dataset=','.join(names['validation']),
    dataset_dir=str(run),
    mix_strategy='interleave_over',
    interleave_probs=','.join(['0.1'] * 10),
    streaming=True,
    accelerator_config={'dispatch_batches': False},
    max_steps=steps_per_epoch * 5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    do_eval=True,
    per_device_eval_batch_size=1,
    prediction_loss_only=True,
    output_dir=str(run / 'training-output'),
    eval_strategy='steps',
    eval_steps=steps_per_epoch,
    save_strategy='steps',
    save_steps=500,
    save_total_limit=5,
    load_best_model_at_end=False,
    report_to='none',
    logging_steps=5,
    dataloader_num_workers=2,
    preprocessing_num_workers=2,
    seed=42,
    buffer_size=4096,
)
config.pop('num_train_epochs', None)
(run / 'training-config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
manifest = dict(
    status='diagnostic_probe_only' if a.diagnostic_probe else 'prepared',
    prompt_version='v5',
    step_m=STEP_M,
    rot_step_deg=ROT_STEP_DEG,
    gripper_steps=GRIPPER_STEPS,
    prompt_sha256=digest(prompt_path),
    episode_splits=splits,
    sample_counts=counts,
    export_sha256=hashes,
    registry_sha256=digest(run / 'dataset_info.json'),
    training_config_sha256=digest(run / 'training-config.yaml'),
    llamafactory_revision=LF_REVISION,
    updates_per_nominal_epoch=steps_per_epoch,
    max_steps=steps_per_epoch * 5,
    nominal_epochs=5,
    checkpoint_selection='evaluate checkpoints at nominal passes 1, 3 and 5 before extending; no loss-only selection',
    benchmark_state_overlap='Not assessed; evaluation starts must not be claimed disjoint from demonstration states',
    chat_template_sha256=digest(root / '.workspace/Show-Harness/models/chat_templates/qwen3_5_nothink.jinja'),
    robot_config_sha256=digest(root / '.workspace/Show-Harness/configs/robot_libero.yaml'),
)
write(run / 'dataset-manifest.json', manifest)
print(
    f'Exported {total} training examples; {steps_per_epoch * 5} updates. Equal task sampling, nominal 5 passes (not exact per-example epochs).'
)
