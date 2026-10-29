"""Measure all translation/rotation tokens in real LIBERO, including repeated targets."""

import argparse
import json
from pathlib import Path
from contract import ControlCounter, ROT_STEP_DEG, STEP_M, executor, import_sim, root_path, write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
root = root_path()
import_sim(root)
a.out.mkdir(parents=True, exist_ok=False)
import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from scripts.run_libero_mvtoken import make_backend_for
from scripts.trajectory.real2sim.atomic_tokenizer import MOVE_DIRS, ROT_AXES, rotation_error, rotvec_to_mat
from libero_boundary_rendering import install_boundary_rendering

suite = benchmark.get_benchmark_dict()['libero_object'](task_order_index=0)
task = suite.get_task(1)
state = suite.get_task_init_states(1)[0]
backend = make_backend_for(
    str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
    'configs/robot_libero.yaml',
)
install_boundary_rendering(backend)
counter = ControlCounter(backend)
rows = []
frames = []


def capture(label):
    av, wr = backend.grab_frames()
    frames.append(Image.fromarray(np.concatenate([av, wr], axis=1)))
    frames[-1].save(a.out / f'{len(frames) - 1:03d}-{label}.png')


try:
    for token in list(MOVE_DIRS) + list(ROT_AXES):
        backend.reset_to_state(state)
        controller = backend.env.env.robots[0].controller
        controller.update(force=True)
        controller.reset_goal()
        backend.set_orientation_ref(backend.tcp_rotmat())
        ex = executor(backend)
        counter.reset()
        capture(token + '-before')
        for repeat in range(2):
            pos = backend.tcp_pos().copy()
            rot = backend.tcp_rotmat().copy()
            before_steps = counter.count
            if token in MOVE_DIRS:
                target = ex.lattice_node(pos) + MOVE_DIRS[token] * STEP_M
                ex.move(token)
                err = float(np.linalg.norm(backend.tcp_pos() - target) * 1000)
                measured = (backend.tcp_pos() - pos) * 1000
                row = dict(
                    token=token,
                    repeat=repeat + 1,
                    commanded_delta_mm=(MOVE_DIRS[token] * 10).tolist(),
                    actual_delta_mm=measured.tolist(),
                    target_error_mm=err,
                    passed=err <= 1.1,
                    actual_angle_deg=float(
                        np.degrees(np.linalg.norm(rotation_error(backend.tcp_rotmat(), rot)))
                    ),
                )
            else:
                target = rotvec_to_mat(ROT_AXES[token] * np.radians(ROT_STEP_DEG)) @ backend.orientation_ref()
                ex.rotate(token)
                err = float(np.degrees(np.linalg.norm(rotation_error(backend.tcp_rotmat(), target))))
                drift = float(np.linalg.norm(backend.tcp_pos() - pos) * 1000)
                row = dict(
                    token=token,
                    repeat=repeat + 1,
                    commanded_angle_deg=ROT_STEP_DEG,
                    actual_angle_deg=float(
                        np.degrees(np.linalg.norm(rotation_error(backend.tcp_rotmat(), rot)))
                    ),
                    orientation_target_error_deg=err,
                    position_drift_mm=drift,
                    passed=err <= 0.6 and drift <= 2,
                )
            row['control_steps'] = counter.count - before_steps
            rows.append(row)
            capture(token + f'-after{repeat + 1}')
    backend.reset_to_state(state)
    ex = executor(backend)
    opened = backend.gripper_width()
    ex.grasp()
    closed = backend.gripper_width()
    capture('GRASP')
    ex.release()
    reopened = backend.gripper_width()
    capture('RELEASE')
    grip = dict(
        open_mm=opened * 1000,
        closed_mm=closed * 1000,
        reopened_mm=reopened * 1000,
        passed=opened - closed > 0.02 and reopened - closed > 0.02,
    )
    # Exercise an actual mid-command limit with a counter below max_ctrl_steps.
    backend.reset_to_state(state)
    ex = executor(backend)
    counter.reset(limit=1)
    from contract import StepLimit

    interrupted = False
    try:
        ex.grasp()
    except StepLimit:
        interrupted = True
    budget_passed = interrupted and counter.count == 1
    counter.reset()
    report = dict(
        status='passed' if all(r['passed'] for r in rows) and grip['passed'] and budget_passed else 'failed',
        step_m=STEP_M,
        rot_step_deg=ROT_STEP_DEG,
        moves_and_rotations=rows,
        gripper=grip,
        mid_command_budget_limit_passed=budget_passed,
        interpretation='Target errors are measured in a free-space initial pose; contact execution may underachieve. Repeat targets use the shared executor lattice and orientation reference.',
    )
    write(a.out / 'control-check.json', report)
    import subprocess

    process = subprocess.Popen(
        [
            'ffmpeg',
            '-y',
            '-loglevel',
            'error',
            '-f',
            'rawvideo',
            '-pix_fmt',
            'rgb24',
            '-s',
            '512x256',
            '-r',
            '4',
            '-i',
            '-',
            '-an',
            '-c:v',
            'libx264',
            '-pix_fmt',
            'yuv420p',
            str(a.out / 'controls.mp4'),
        ],
        stdin=subprocess.PIPE,
    )
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    assert process.wait() == 0
    print(json.dumps(report, indent=2))
    assert report['status'] == 'passed'
finally:
    backend.close()
