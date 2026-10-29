"""One video-recorded rollout under the frozen 1 cm / 5 degree contract.

The run's dataset-manifest.json selects the prompt (v5, or a variant with a robot-state line)
and the recent-moves window. --turn-guard-deg is a DIAGNOSTIC option outside the frozen
benchmark contract; see ../docs/turn-guard.md before using or reporting it.
"""

import argparse
import json
from pathlib import Path
import time
from contract import (
    ControlCounter,
    GRIPPER_STEPS,
    ROT_STEP_DEG,
    STEP_M,
    StepLimit,
    digest,
    executor,
    history_window,
    import_sim,
    prompt_file,
    robot_state_text,
    served_adapter_name,
    root_path,
    turn_guard_blocked,
    turned_since_start,
    write,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--task', type=int, required=True)
p.add_argument('--initial-state', type=int, required=True)
p.add_argument('--url', default='http://127.0.0.1:8000/v1')
p.add_argument('--budget', choices=['matched', 'native'], default='matched')
p.add_argument(
    '--initial-state-file',
    type=Path,
    help='Reserved demonstration start for validation, not benchmark scoring',
)
p.add_argument(
    '--history',
    choices=['normal', 'none', 'no-rt'],
    default='normal',
    help='Diagnostic only: normal = frozen contract; none = always "none"; no-rt = drop RT_* tokens from recent moves',
)
p.add_argument(
    '--turn-guard-deg',
    type=float,
    default=0,
    help='DIAGNOSTIC, not the frozen benchmark: withhold the yaw token that would turn past this net yaw '
    '(e.g. 90). 0 = off. See ../docs/turn-guard.md',
)
a = p.parse_args()
root = root_path()
import_sim(root)
a.out.mkdir(parents=True, exist_ok=False)
assert a.turn_guard_deg >= 0
import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from core.vlm.vlm_client import VLMClient
from core.vlm.mvtoken_roles import MvTokenController
from scripts.run_libero_mvtoken import bddl_language, make_backend_for
from scripts.trajectory.real2sim.atomic_tokenizer import RolloutWriter, token_kind
from libero_boundary_rendering import install_boundary_rendering

manifest = json.loads((a.run / 'dataset-manifest.json').read_text())
assert manifest['step_m'] == STEP_M and manifest['rot_step_deg'] == ROT_STEP_DEG
prompt = prompt_file(a.run, manifest)
assert digest(prompt) == manifest['prompt_sha256']
window = history_window(manifest)
state_line = bool(manifest.get('state_line'))
suite = benchmark.get_benchmark_dict()['libero_object'](task_order_index=0)
task = suite.get_task(a.task)
bddl = str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file)
states = suite.get_task_init_states(a.task)
state = np.load(a.initial_state_file) if a.initial_state_file else states[a.initial_state]
backend = make_backend_for(bddl, 'configs/robot_libero.yaml')
backend.settle_steps = 10
install_boundary_rendering(backend)
counter = ControlCounter(backend)
client = VLMClient(
    base_url=a.url,
    model=served_adapter_name(manifest),
    api_key='EMPTY',
    timeout_s=120,
    max_tokens=16,
    temperature=0,
)
controller = MvTokenController(client, prompt.read_text())
writer = RolloutWriter(a.out / 'rollout')
decisions = []
tokens = []
recent = []
stuck = {}
reason = 'max_tokens'
partial = False
started = time.monotonic()
guard_decisions = 0
max_abs_yaw = 0.0
write(
    a.out / 'resolved-settings.json',
    dict(
        task=a.task,
        initial_state=a.initial_state,
        prompt_sha256=digest(prompt),
        prompt_file=prompt.name,
        history_window=window,
        state_line=state_line,
        step_m=STEP_M,
        rot_step_deg=ROT_STEP_DEG,
        gripper_steps=GRIPPER_STEPS,
        budget=a.budget,
        max_control_steps=280 if a.budget == 'matched' else None,
        max_tokens=400,
        settle_steps=10,
        validation=bool(a.initial_state_file),
        stuck_guard_mm=2.5,
        history=a.history,
        turn_guard_deg=a.turn_guard_deg or None,
        protocol='frozen' if not a.turn_guard_deg and a.history == 'normal' else 'diagnostic',
    ),
)
try:
    backend.reset_to_state(state)
    np.save(a.out / 'provided-initial-state.npy', state)
    counter.reset(limit=280 if a.budget == 'matched' else None)
    ex = executor(backend)
    start_pose = backend.tcp_pose7()
    for index in range(400):
        if backend.success():
            reason = 'success'
            break
        if counter.limit is not None and counter.count >= counter.limit:
            reason = 'control_step_limit'
            break
        av, wr = backend.grab_frames()
        turned = turned_since_start(start_pose, backend.tcp_pose7())
        max_abs_yaw = max(max_abs_yaw, abs(turned[2]))
        guarded = turn_guard_blocked(turned, a.turn_guard_deg)
        guard_decisions += bool(guarded)
        blocked = {t for t, n in stuck.items() if n >= 2} | guarded
        robot_state = robot_state_text(ex.gripper_closed, turned) if state_line else ''
        answer = controller.decide(
            task=bddl_language(bddl),
            gripper_state=robot_state,
            recent_moves=', '.join(recent) or 'none',
            agentview_image=av,
            wrist_image=wr,
            constrain_to=[t for t in controller.allowed_tokens if t not in blocked] if blocked else None,
        )
        token = answer.token
        decisions.append(
            dict(
                index=index,
                token=token,
                raw=answer.raw_text,
                control_steps=counter.count,
                turned_deg=[round(v, 1) for v in turned],
                robot_state=robot_state or None,
                turn_guard_withheld=sorted(guarded) or None,
                blocked=sorted(blocked) or None,
            )
        )
        write(a.out / 'decisions.json', decisions)
        if token == 'DONE':
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
        before = backend.tcp_pos().copy()
        tokens.append(token)
        try:
            if token.startswith('MV_'):
                ex.move(token)
            elif token.startswith('RT_'):
                ex.rotate(token)
            elif token == 'GRASP':
                ex.grasp()
            elif token == 'RELEASE':
                ex.release()
            else:
                raise ValueError(f'Invalid action {token}')
        except StepLimit:
            reason = 'control_step_limit'
            partial = True
            break
        if token.startswith('MV_'):
            if np.linalg.norm(backend.tcp_pos() - before) * 1000 < 2.5:
                stuck[token] = stuck.get(token, 0) + 1
            else:
                stuck.clear()
        if (
            token.startswith(('MV_', 'RT_'))
            and a.history != 'none'
            and not (a.history == 'no-rt' and token.startswith('RT_'))
        ):
            recent.insert(0, token)
            recent = recent[:window]
        print('decision', len(decisions), token, 'control steps', counter.count, flush=True)
except BaseException as e:
    reason = 'error:' + type(e).__name__
    write(a.out / 'error.json', dict(error=str(e)))
    raise
finally:
    if backend._obs is not None:
        av, wr = backend.grab_frames()
        Image.fromarray(av).save(a.out / 'final-agentview.png')
        Image.fromarray(wr).save(a.out / 'final-wrist.png')
        episode = dict(
            task=bddl_language(bddl),
            success=backend.success(),
            reason=reason,
            num_steps=len(tokens),
            tokens=tokens,
            control_steps=counter.count,
            partial_last_action=partial,
            wall_s=time.monotonic() - started,
            turn_guard_deg=a.turn_guard_deg or None,
            turn_guard_decisions=guard_decisions,
            max_abs_yaw_deg=round(max_abs_yaw, 1),
        )
        writer.close(dict(episode, step_m=STEP_M, rot_step_deg=ROT_STEP_DEG, gripper_steps=GRIPPER_STEPS))
        write(
            a.out / 'smoke-check.json',
            dict(
                status='completed' if not reason.startswith('error:') else 'error',
                episode=episode,
                decisions=len(decisions),
            ),
        )
        np.save(a.out / 'final-state.npy', backend.sim_state())
    backend.close()
