"""Export a training variant from an already prepared v5 run, without reconverting demonstrations.

Default variant (see ../docs/state-and-changepoints.md):
  * robot-state line: gripper open/closed and net turn since the start of the attempt, measured
    from the logged TCP pose with contract.turned_since_start (rollouts use the same function);
  * shorter recent-moves window (3 instead of Show-Harness's 5);
  * action-changepoint oversampling in the TRAIN split only: every sample whose label differs
    from the previous label, plus GRASP/RELEASE/DONE, appears --changepoint-repeat times.

The episode-level train/validation split is copied from the source run, so validation
episodes stay held out and comparable. Images are referenced in place in the source run.
The update budget per nominal pass is computed from the un-oversampled training count, so
``--passes N`` costs the same number of updates as the v5 run.
"""

import argparse
import collections
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import yaml
from contract import (
    ROT_STEP_DEG,
    STEP_M,
    V5_PROMPT_FILE,
    digest,
    nominal_turn,
    robot_state_text,
    root_path,
    turned_since_start,
    write,
)

ALLOWED = {
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
ALWAYS_CHANGEPOINT = {'GRASP', 'RELEASE', 'DONE'}
# Base model -> local weights, serve_qwen.sh profile and the name the trained LoRA is served under.
BASE_MODELS = {
    '4b': dict(
        weights='.workspace/models-4b/base',
        serve_profile='4b-libero-v5',
        served_adapter_name='qwen3_5_4b_libero_v5',
        hf_name='Qwen/Qwen3.5-4B',
    ),
    '2b': dict(
        weights='.workspace/models/base',
        serve_profile='2b-libero',
        served_adapter_name='qwen3_5_2b_libero',
        hf_name='Qwen/Qwen3.5-2B',
    ),
}

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument(
    '--source', type=Path, required=True, help='Prepared v5 run (dataset-manifest.json status=prepared)'
)
p.add_argument(
    '--out',
    type=Path,
    help='New run directory (default: .workspace/libero-object-1cm5deg-<variant>-<timestamp>)',
)
p.add_argument(
    '--history-window', type=int, default=3, help='Recent MV_/RT_ tokens listed in the prompt (v5: 5)'
)
p.add_argument('--no-state-line', action='store_true', help='Keep the v5 prompt text (no robot-state line)')
p.add_argument(
    '--changepoint-repeat', type=int, default=4, help='Total copies of each changepoint TRAIN sample; 1 = off'
)
p.add_argument('--seed', type=int, default=42)
p.add_argument(
    '--base-model',
    choices=sorted(BASE_MODELS),
    default='4b',
    help='Base model to fine-tune; 2b = Qwen3.5-2B, the Show-Harness default backbone (same data and LoRA recipe)',
)
a = p.parse_args()
root = root_path()
source = a.source.resolve()
assert 0 <= a.history_window <= 20 and a.changepoint_repeat >= 1
state_line = not a.no_state_line
variant = (
    ('v6-state' if state_line else 'v5')
    + f'-h{a.history_window}-cp{a.changepoint_repeat}'
    + ('' if a.base_model == '4b' else '-' + a.base_model)
)
model = BASE_MODELS[a.base_model]
assert (root / model['weights'] / 'config.json').is_file(), f"Missing base weights {model['weights']}"
assert variant != 'v5-h5-cp1', 'That is the source v5 export; nothing to vary'
out = (
    a.out
    or root
    / '.workspace'
    / f"libero-object-1cm5deg-{variant}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
).resolve()
out.mkdir(parents=True, exist_ok=False)

src_manifest = json.loads((source / 'dataset-manifest.json').read_text())
assert src_manifest['status'] == 'prepared' and src_manifest.get('prompt_version') == 'v5'
assert src_manifest['step_m'] == STEP_M and src_manifest['rot_step_deg'] == ROT_STEP_DEG
v5_prompt = source / V5_PROMPT_FILE
assert digest(v5_prompt) == src_manifest['prompt_sha256']
template = v5_prompt.read_text()
if state_line:
    history_line = 'Recent moves, newest first: {recent_moves}'
    step_line = '- Each MV_* token moves the gripper 1 cm; each RT_* token turns it 5 degrees'
    assert (
        template.count(history_line) == 1
        and template.count(step_line) == 1
        and '{gripper_state}' not in template
    )
    template = template.replace(history_line, 'Robot state: {gripper_state}\n' + history_line)
    template = template.replace(
        step_line,
        step_line
        + '\n- Robot state gives the gripper (open/closed) and its net turn since the start of the task',
    )
    prompt_name = 'prompt-v6-state-1cm-5deg.txt'
else:
    prompt_name = V5_PROMPT_FILE
(out / prompt_name).write_text(template)
(out / 'tasks.json').write_text((source / 'tasks.json').read_text())

rollouts = {
    (row['task'], row['demo']): Path(row['rollout'])
    for row in json.loads((source / 'preparation-report.json').read_text())['episodes']
    if row['success']
}


def episode_samples(directory, task_name):
    meta = json.loads((directory / 'metadata.json').read_text())
    assert meta['success'] and meta['step_m'] == STEP_M and meta['rot_step_deg'] == ROT_STEP_DEG
    assert meta['prompt_sha256'] == src_manifest['prompt_sha256'] and meta['task_key'] == task_name
    records = [json.loads(line) for line in (directory / 'actions.jsonl').read_text().splitlines()]
    start = records[0]['ee_pose']
    history = []
    samples = []
    previous = None
    last = records[-1]
    last_turned = turned_since_start(start, last['ee_pose'])
    # The DONE frame has no logged pose: take the last pose plus the last token's ideal effect.
    final = dict(
        meta['final_frame'],
        token='DONE',
        gripper_closed={'GRASP': True, 'RELEASE': False}.get(last['token'], last['gripper_closed']),
        turned=nominal_turn(last_turned, last['token']) if last['token'].startswith('RT_') else last_turned,
    )
    for record in records + [final]:
        token = record['token']
        assert token in ALLOWED
        images = [str((directory / record[key]).resolve()) for key in ('agentview', 'wrist')]
        assert all(Path(image).is_file() for image in images)
        turned = record['turned'] if 'turned' in record else turned_since_start(start, record['ee_pose'])
        fields = dict(
            task=meta['task'],
            recent_moves=', '.join(history[-a.history_window :][::-1]) if a.history_window else '',
        )
        fields['recent_moves'] = fields['recent_moves'] or 'none'
        if state_line:
            fields['gripper_state'] = robot_state_text(record['gripper_closed'], turned)
        changepoint = previous is None or token != previous or token in ALWAYS_CHANGEPOINT
        samples.append(
            (
                dict(
                    instruction='<image><image>' + template.format(**fields),
                    input='',
                    output=token,
                    images=images,
                ),
                changepoint,
            )
        )
        previous = token
        if token.startswith(('MV_', 'RT_')):
            history.append(token)
    return samples


exports = {'train': {}, 'validation': {}}
counts = {}
stats = {}
hashes = {}
registry = {}
names = {'train': [], 'validation': []}
for task in json.loads((source / 'tasks.json').read_text()):
    t = task['index']
    counts[str(t)] = {}
    stats[str(t)] = {}
    for split in ('train', 'validation'):
        samples = []
        for demo in src_manifest['episode_splits'][str(t)][split]:
            samples += episode_samples(rollouts[(t, demo)], task['name'])
        n_cp = sum(cp for _, cp in samples)
        base = len(samples)
        if split == 'train':
            rows = [s for s, cp in samples for _ in range(a.changepoint_repeat if cp else 1)]
            random.Random(a.seed + t).shuffle(rows)  # spread copies across the streaming shuffle buffer
        else:
            rows = [s for s, _ in samples]  # episode order kept for check_changepoints.py
        stats[str(t)][split] = dict(
            samples=base,
            changepoints=n_cp,
            changepoint_fraction=round(n_cp / base, 4),
            rows_written=len(rows),
            changepoint_fraction_written=round(
                n_cp * (a.changepoint_repeat if split == 'train' else 1) / len(rows), 4
            ),
        )
        counts[str(t)][split] = base
        filename = f'{split}-task{t:02d}.json'
        write(out / filename, rows)
        hashes[filename] = digest(out / filename)
        name = f"object_{variant.replace('-', '_')}_{split}_{t:02d}"
        names[split].append(name)
        registry[name] = dict(
            file_name=filename,
            columns=dict(prompt='instruction', query='input', response='output', images='images'),
        )
write(out / 'dataset_info.json', registry)

total = sum(counts[t]['train'] for t in counts)
steps_per_epoch = math.ceil(total / 32)
config = yaml.safe_load((source / 'training-config.yaml').read_text())
config.update(
    model_name_or_path=str(root / model['weights']),
    dataset=','.join(names['train']),
    eval_dataset=','.join(names['validation']),
    dataset_dir=str(out),
    output_dir=str(out / 'training-output'),
    max_steps=steps_per_epoch * 5,
    eval_steps=steps_per_epoch,
)
(out / 'training-config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
manifest = dict(
    src_manifest,
    status='prepared',
    prompt_version='v6-state' if state_line else 'v5',
    prompt_file=prompt_name,
    prompt_sha256=digest(out / prompt_name),
    history_window=a.history_window,
    state_line=state_line,
    state_definition='contract.robot_state_text(gripper_closed, contract.turned_since_start(first_pose, pose)); '
    'DONE frame: last pose + nominal effect of the last token'
    if state_line
    else None,
    changepoint_repeat=a.changepoint_repeat,
    changepoint_rule='label differs from previous label in the episode, '
    'or label is GRASP/RELEASE/DONE; train split only; rows shuffled with seed+task',
    changepoint_stats=stats,
    sample_counts=counts,
    export_sha256=hashes,
    registry_sha256=digest(out / 'dataset_info.json'),
    training_config_sha256=digest(out / 'training-config.yaml'),
    updates_per_nominal_epoch=steps_per_epoch,
    max_steps=steps_per_epoch * 5,
    nominal_epoch_definition='un-oversampled training examples / 32, same budget as v5',
    variant=variant,
    base_model=model['hf_name'],
    serve_profile=model['serve_profile'],
    served_adapter_name=model['served_adapter_name'],
    source_run=str(source),
    source_manifest_sha256=digest(source / 'dataset-manifest.json'),
)
write(out / 'dataset-manifest.json', manifest)
all_cp = sum(s['train']['changepoints'] for s in stats.values())
all_n = sum(s['train']['samples'] for s in stats.values())
print(
    f'Variant {variant} -> {out}\nTrain examples {all_n} (changepoints {all_cp}, {all_cp / all_n:.1%}; '
    f'{a.changepoint_repeat}x -> {all_cp * a.changepoint_repeat / (all_n + all_cp * (a.changepoint_repeat - 1)):.1%} of rows). '
    f'{steps_per_epoch} updates per nominal pass.'
)
