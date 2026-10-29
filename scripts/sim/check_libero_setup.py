"""Freeze the ten LIBERO task files and render one initial scene, without a model."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show-harness', type=Path, required=True)
    parser.add_argument('--libero', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sh, libero, out = (p.resolve() for p in (args.show_harness, args.libero, args.out))
    sys.path.insert(0, str(sh))
    from libero.libero import benchmark, get_libero_path
    from core.config import camera_contract, load_yaml
    from scripts.trajectory.real2sim.backends.libero import LiberoBackend

    out.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0)
    assert suite.n_tasks == 10
    tasks = []
    for i in range(suite.n_tasks):
        task = suite.get_task(i)
        bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        init = Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file
        states = suite.get_task_init_states(i)
        tasks.append(
            {
                'name': Path(task.bddl_file).stem,
                'bddl': str(bddl.relative_to(libero)),
                'bddl_sha256': sha256(bddl),
                'initial_states': str(init.relative_to(libero)),
                'initial_states_sha256': sha256(init),
                'initial_state_count': len(states),
                'initial_state_index': 0,
            }
        )
    revisions = {
        name: subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        for name, path in [('show_harness', sh), ('libero', libero)]
    }
    (out / 'reference.json').write_text(
        json.dumps(
            {
                'revisions': revisions,
                'suite': 'libero_spatial',
                'task_order_index': 0,
                'seed': 0,
                'episodes_per_task': 1,
                'tasks': tasks,
            },
            indent=2,
        )
        + '\n'
    )
    config = load_yaml(sh / 'configs/robot_libero.yaml')
    task = suite.get_task(0)
    bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    backend = LiberoBackend.make(str(bddl), camera_size=256, **camera_contract(config))
    try:
        backend.reset_to_state(np.asarray(suite.get_task_init_states(0)[0]))
        agent, wrist = backend.grab_frames()
        stats = {}
        for name, pixels in [('agentview', agent), ('wrist', wrist)]:
            assert pixels.shape == (256, 256, 3), (name, pixels.shape)
            assert pixels.dtype == np.uint8 and float(pixels.std()) > 1, name
            Image.fromarray(pixels).save(out / f'{name}.png')
            stats[name] = {
                'shape': list(pixels.shape),
                'dtype': str(pixels.dtype),
                'pixel_std': float(pixels.std()),
            }
        np.save(out / 'scene_state.npy', backend.sim_state())
        (out / 'render-check.json').write_text(
            json.dumps(
                {
                    'task': tasks[0]['name'],
                    'initial_state_index': 0,
                    'seed': 0,
                    'render_backend': 'osmesa',
                    'camera_configuration': config,
                    'fixture_poses': backend.fixture_poses(),
                    'image_checks': stats,
                    'model_queried': False,
                },
                indent=2,
            )
            + '\n'
        )
        print('PASS: all 10 task definitions and initial-state files loaded; both camera views rendered.')
        print('Artifacts:', out)
    finally:
        backend.env.close()


if __name__ == '__main__':
    main()
