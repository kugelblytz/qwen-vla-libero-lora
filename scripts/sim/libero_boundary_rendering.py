"""Defer unused camera renders while retaining the pinned upstream control loop.

Robot observables and all physics steps still update normally. Camera sampler clocks
advance normally, but RGB computation happens only when grab_frames() is requested.
Fresh boundary observations differ slightly from the upstream camera sampling time.
"""

import functools
import time
import numpy as np


def install_boundary_rendering(backend):
    if getattr(backend, 'boundary_rendering', None) is not None:
        raise RuntimeError('Boundary rendering is already installed')
    original_reset = backend.reset_to_state
    original_grab = backend.grab_frames
    sensors = {}
    latest = {}
    stats = {'mode': 'boundary', 'renders': 0, 'skipped_renders': 0, 'render_s': 0.0}
    backend.boundary_rendering = stats

    def reset(state):
        original_reset(state)
        # LIBERO hard_reset replaces observables. Rebind after every reset.
        env = backend.env.env
        sensors.clear()
        latest.clear()
        stats.update(renders=0, skipped_renders=0, render_s=0.0)
        for camera in [backend.agentview_camera, backend.wrist_camera]:
            name = camera + '_image'
            observable = env._observables[name]
            assert observable.modality == 'image' and observable.is_enabled()
            sensors[name] = observable._sensor
            latest[name] = np.asarray(backend._obs[name]).copy()

            def make_deferred(name, original):
                @functools.wraps(original)
                def deferred(cache):
                    stats['skipped_renders'] += 1
                    return latest[name]

                return deferred

            observable._sensor = make_deferred(name, sensors[name])

    def grab():
        if not sensors:
            raise RuntimeError('grab_frames() before reset_to_state()')
        for name, sensor in sensors.items():
            start = time.perf_counter()
            image = sensor(backend.env.env._obs_cache)
            stats['render_s'] += time.perf_counter() - start
            stats['renders'] += 1
            latest[name] = image
            backend._obs[name] = image
        return original_grab()  # Preserve camera transforms and sim-state recording.

    backend.reset_to_state = reset
    backend.grab_frames = grab
    return stats
