"""Encode both camera streams and the final observation at four decisions/second."""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('episode', type=Path)
    args = ap.parse_args()
    root = args.episode.resolve()
    assert shutil.which('ffmpeg'), 'Install ffmpeg to produce rollout videos'
    agent = sorted((root / 'rollout/agentview').glob('*.png'))
    wrist = sorted((root / 'rollout/wrist').glob('*.png'))
    assert [p.name for p in agent] == [p.name for p in wrist], 'Camera frame streams differ'
    if (root / 'final-agentview.png').exists() and (root / 'final-wrist.png').exists():
        agent.append(root / 'final-agentview.png')
        wrist.append(root / 'final-wrist.png')
    assert agent, 'No observations available for this attempt; inspect error.json'
    with tempfile.TemporaryDirectory(prefix='rollout-video-') as tmp:
        work = Path(tmp)
        for name, frames in [('agent', agent), ('wrist', wrist)]:
            dest = work / name
            dest.mkdir()
            for i, frame in enumerate(frames):
                (dest / f'{i:04d}.png').symlink_to(frame)
        subprocess.run(
            [
                'ffmpeg',
                '-nostdin',
                '-hide_banner',
                '-loglevel',
                'error',
                '-framerate',
                '4',
                '-i',
                str(work / 'agent/%04d.png'),
                '-framerate',
                '4',
                '-i',
                str(work / 'wrist/%04d.png'),
                '-filter_complex',
                'hstack',
                '-c:v',
                'libx264',
                '-pix_fmt',
                'yuv420p',
                '-movflags',
                '+faststart',
                str(root / 'trajectory.mp4'),
            ],
            check=True,
        )
    print(f'Video: {root / "trajectory.mp4"} ({len(agent)} frames; agent left, wrist right)')


if __name__ == '__main__':
    main()
