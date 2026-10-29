"""Local pre-grasp diagnostic; expert control and model share one held-out state."""

import argparse
import json
from pathlib import Path
import time
from contract import digest, executor, import_sim, root_path, write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--case', type=Path, required=True)
p.add_argument('--mode', choices=['expert', 'model'], required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--url', required=True)
a = p.parse_args()
case = json.loads(a.case.read_text())
root = root_path()
import_sim(root)
import numpy as np
from PIL import Image
from core.vlm.vlm_client import VLMClient
from core.vlm.mvtoken_roles import MvTokenController
from scripts.run_libero_mvtoken import make_backend_for
from scripts.trajectory.real2sim.atomic_tokenizer import RolloutWriter, token_kind
from libero_boundary_rendering import install_boundary_rendering

run = Path(case['run'])
directory = Path(case['rollout'])
manifest = json.loads((run / 'dataset-manifest.json').read_text())
prompt = run / 'prompt-v5-1cm-5deg.txt'
assert digest(prompt) == manifest['prompt_sha256']
assert case['demo'] in manifest['episode_splits'][str(case['task'])]['validation']
meta = json.loads((directory / 'metadata.json').read_text())
records = [json.loads(s) for s in (directory / 'actions.jsonl').read_text().splitlines()]
index = next(i for i, r in enumerate(records) if r['token'] == 'GRASP')
states = np.load(directory / 'states.npz')['states']
backend = make_backend_for(meta['bddl'], 'configs/robot_libero.yaml')
backend.settle_steps = 0
install_boundary_rendering(backend)
origin = backend.pose_from_state(states[0])[0].copy()
backend.reset_to_state(states[index])
ex = executor(backend)
ex.lattice_origin = origin
target = meta['task_key'].removeprefix('pick_up_the_').removesuffix('_and_place_it_in_the_basket') + '_1'
assert target in backend.env.env.obj_body_id, target
obj = backend.env.env.get_object(target)
z0 = float(backend.object_pos(target)[2])
recent = [r['token'] for r in records[:index] if r['token'].startswith(('MV_', 'RT_'))][-5:][::-1]
controller = MvTokenController(
    VLMClient(
        base_url=a.url,
        model='qwen3_5_4b_libero_v5',
        api_key='EMPTY',
        timeout_s=120,
        max_tokens=16,
        temperature=0,
    ),
    prompt.read_text(),
)
a.out.mkdir(parents=True, exist_ok=False)
writer = RolloutWriter(a.out / 'rollout')
rows = []
sustained = 0
reason = 'decision_limit'
started = time.monotonic()


def measurement():
    return dict(
        target_z=float(backend.object_pos(target)[2]),
        lift_m=float(backend.object_pos(target)[2]) - z0,
        finger_contact=bool(backend.env.env._check_grasp(backend._robot.gripper, obj.contact_geoms)),
        width_m=backend.gripper_width(),
        tcp=backend.tcp_pos().tolist(),
    )


initial = measurement()
try:
    for step in range(24):
        av, wr = backend.grab_frames()
        if a.mode == 'expert':
            token = records[index + step]['token']
            raw = token
        else:
            answer = controller.decide(
                task=meta['task'],
                gripper_state='',
                recent_moves=', '.join(recent) or 'none',
                agentview_image=av,
                wrist_image=wr,
            )
            token = answer.token
            raw = answer.raw_text
        if token == 'DONE':
            rows.append(dict(index=step, token=token, raw=raw, **measurement()))
            reason = 'done'
            break
        writer.add_step(
            token=token,
            kind=token_kind(token),
            agentview=av,
            wrist=wr,
            gripper_closed=ex.gripper_closed,
            ee_pose=backend.tcp_pose7(),
            width=backend.gripper_width(),
        )
        if token.startswith('MV_'):
            ex.move(token)
        elif token.startswith('RT_'):
            ex.rotate(token)
        elif token == 'GRASP':
            ex.grasp()
        elif token == 'RELEASE':
            ex.release()
        else:
            raise ValueError(token)
        row = dict(index=step, token=token, raw=raw, **measurement())
        rows.append(row)
        supported = row['lift_m'] >= 0.03 and row['finger_contact'] and ex.gripper_closed
        sustained = sustained + 1 if supported else 0
        if token.startswith(('MV_', 'RT_')):
            recent = [token, *recent][:5]
        write(a.out / 'measurements.json', rows)
        print(
            a.mode,
            step,
            token,
            'lift_mm',
            round(row['lift_m'] * 1000, 1),
            'contact',
            row['finger_contact'],
            flush=True,
        )
        if sustained >= 3:
            reason = 'supported_lift'
            break
except BaseException as e:
    reason = 'error:' + type(e).__name__
    write(a.out / 'error.json', dict(error=str(e)))
    raise
finally:
    av, wr = backend.grab_frames()
    Image.fromarray(av).save(a.out / 'final-agentview.png')
    Image.fromarray(wr).save(a.out / 'final-wrist.png')
    result = dict(
        task=case['task'],
        demo=case['demo'],
        target=target,
        mode=a.mode,
        reason=reason,
        grasp_and_lift_pass=reason == 'supported_lift',
        initial=initial,
        max_lift_m=max([0.0, *[r['lift_m'] for r in rows]]),
        any_finger_contact=any(r['finger_contact'] for r in rows),
        predicted_grasp=any(r['token'] == 'GRASP' for r in rows),
        tokens=[r['token'] for r in rows],
        decisions=len(rows),
        wall_s=time.monotonic() - started,
        criterion='Target lifted >=3cm with both finger-pad contacts for three consecutive action boundaries.',
        caveat='Held-out pre-grasp state diagnostic, not an approach test or benchmark success.',
    )
    writer.close(result)
    write(a.out / 'result.json', result)
    write(a.out / 'measurements.json', rows)
    backend.close()
