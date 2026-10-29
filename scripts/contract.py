"""One action contract for preparation, training and rollout evaluation."""

import hashlib
import json
import os
from pathlib import Path
import platform

STEP_M = 0.01
ROT_STEP_DEG = 5.0
ROT_TOL_DEG = 3.0
GRIPPER_STEPS = 25
MAX_TOKENS = 600
SHOW_REVISION = 'e9867e971c3a8c5da352e151006cac859923e85e'
LF_REVISION = '9ce6b663e9d87cd3c0cb42a1d3ff5cdfe292426d'


def root_path():
    return Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def freeze_prompt(root, run):
    original = (root / '.workspace/Show-Harness/prompts/v5/mvtoken_generator_lite.txt').read_text()
    assert original.count('2 cm') == 1 and original.count('10 degrees') == 1
    prompt = original.replace('2 cm', '1 cm').replace('10 degrees', '5 degrees')
    path = run / 'prompt-v5-1cm-5deg.txt'
    if path.exists():
        assert path.read_text() == prompt, 'Frozen prompt mismatch'
    else:
        path.write_text(prompt)
    return path


def simulator_env(root):
    env = dict(os.environ, MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa', OMP_NUM_THREADS='4')
    libs = root / f'.workspace/system-libs/root/usr/lib/{platform.machine()}-linux-gnu'
    if libs.is_dir():
        env['LD_LIBRARY_PATH'] = str(libs) + ':' + env.get('LD_LIBRARY_PATH', '')
    return env


def import_sim(root):
    import sys

    sys.path.insert(0, str(root / '.workspace/Show-Harness'))
    sys.path.insert(0, str(root / 'scripts/sim'))


def executor(backend):
    import numpy as np
    from scripts.trajectory.real2sim.atomic_tokenizer import AtomicExec

    return AtomicExec(
        backend,
        step_m=STEP_M,
        max_cmd_m=STEP_M,
        max_ctrl_steps=24,
        gripper_steps=GRIPPER_STEPS,
        rot_step_rad=np.radians(ROT_STEP_DEG),
        max_cmd_rad=np.radians(ROT_STEP_DEG),
        lattice_origin=backend.tcp_pos(),
    )


class StepLimit(RuntimeError):
    pass


# ---------------------------------------------------------------- prompt / history
V5_PROMPT_FILE = 'prompt-v5-1cm-5deg.txt'
V5_HISTORY_WINDOW = 5


def prompt_file(run, manifest):
    """The run's frozen prompt. v5 runs predate the manifest field and use the v5 name."""
    return Path(run) / manifest.get('prompt_file', V5_PROMPT_FILE)


def history_window(manifest):
    """How many recent MV_/RT_ tokens the prompt lists (Show-Harness default: 5)."""
    return int(manifest.get('history_window', V5_HISTORY_WINDOW))


# Base model / serving. 4B runs predate these manifest fields and keep the original values.
def serve_profile(manifest):
    """QWEN_PROFILE for serve_qwen.sh (4B: 4b-libero-v5, 2B: 2b-libero)."""
    return manifest.get('serve_profile', '4b-libero-v5')


def served_adapter_name(manifest):
    """Model id under which serve_qwen.sh registers the trained LoRA."""
    return manifest.get('served_adapter_name', 'qwen3_5_4b_libero_v5')


# ------------------------------------------------------------ turned-since-start state
# Show-Harness RT_* tokens turn about WORLD axes through the TCP (atomic_tokenizer.ROT_AXES):
# RT_ROLL_RIGHT=+X, RT_PITCH_BACK=+Y, RT_YAW_CCW=+Z. ee_pose is [x, y, z, qw, qx, qy, qz]
# (backend.tcp_pose7(); a downward gripper logs as ~[0, 1, 0, 0]). The net turn since the
# start of an attempt is the world-frame rotation vector of R_now @ R_start.T, so a pure
# run of RT_YAW_CW tokens reads as yaw = -5 deg per token. Training export and rollout
# evaluation both call these functions, so the state the model sees is defined once.
ROT_WORDS = (('roll', 'RIGHT', 'LEFT'), ('pitch', 'BACK', 'FWD'), ('yaw', 'CCW', 'CW'))


def pose_rotation(pose7):
    import numpy as np

    w, x, y, z = (float(v) for v in pose7[3:7])
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def turned_since_start(start_pose7, pose7):
    """Net world-frame turn [roll, pitch, yaw] in degrees from the attempt's first pose."""
    import numpy as np

    r = pose_rotation(pose7) @ pose_rotation(start_pose7).T
    angle = float(np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0)))
    if angle < 1e-9:
        return [0.0, 0.0, 0.0]
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if np.linalg.norm(axis) < 1e-6:  # ~180 deg: take the axis from the symmetric part
        b = (r + np.eye(3)) / 2
        i = int(np.argmax(np.diag(b)))
        axis = b[:, i]
    return [float(v) for v in np.degrees(axis / np.linalg.norm(axis) * angle)]


def nominal_turn(turned, token):
    """``turned`` after one ideal RT_* token (used only for the DONE frame, which has no pose)."""
    turned = list(turned)
    for i, (_, positive, negative) in enumerate(ROT_WORDS):
        if token == f'RT_{ROT_WORDS[i][0].upper()}_{positive}':
            turned[i] += ROT_STEP_DEG
        elif token == f'RT_{ROT_WORDS[i][0].upper()}_{negative}':
            turned[i] -= ROT_STEP_DEG
    return turned


def robot_state_text(gripper_closed, turned):
    """Prompt line content, rounded to whole token steps so it is stable frame to frame."""
    parts = []
    for (name, positive, negative), value in zip(reversed(ROT_WORDS), reversed(turned)):
        steps = round(value / ROT_STEP_DEG)
        parts.append(
            f'{name} 0'
            if steps == 0
            else f'{name} {abs(steps) * ROT_STEP_DEG:g} {positive if steps > 0 else negative}'
        )
    return f"Gripper: {'closed' if gripper_closed else 'open'}. Turned since start: {', '.join(parts)}."


# ------------------------------------------------------------------------ turn guard
# DIAGNOSTIC EVALUATION OPTION, NOT PART OF THE FROZEN BENCHMARK CONTRACT. See ../docs/turn-guard.md.
# In the checkpoint-3464 native evaluation, every all-RT_YAW_CW attempt drove the arm away
# from its position after 26-27 tokens (~125-130 deg of yaw): the controller can no longer
# hold the TCP and the attempt is unrecoverable. Expert demonstrations never exceed 70 deg
# of net yaw. When the measured net yaw reaches the guard limit, the yaw token that would
# turn further is withheld from that decision (guided decoding over the remaining tokens,
# the same mechanism as the MV_* stuck guard). Turning back is always allowed.
def turn_guard_blocked(turned, limit_deg):
    if not limit_deg:
        return set()
    yaw = turned[2]
    if yaw <= -limit_deg:
        return {'RT_YAW_CW'}
    if yaw >= limit_deg:
        return {'RT_YAW_CCW'}
    return set()


class ControlCounter:
    """Count env.step calls, and interrupt even inside an atomic action."""

    def __init__(self, backend):
        self.count = 0
        self.limit = None
        original = backend.env.step

        def step(*a, **kw):
            if self.limit is not None and self.count >= self.limit:
                raise StepLimit('Simulator control-step limit reached')
            result = original(*a, **kw)
            self.count += 1
            return result

        backend.env.step = step

    def reset(self, limit=None):
        self.count = 0
        self.limit = limit
