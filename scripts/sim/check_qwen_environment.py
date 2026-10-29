"""Verify serving imports and actual CUDA execution without downloading model weights."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': {
            p: importlib.metadata.version(p) for p in ['vllm', 'torch', 'transformers', 'huggingface-hub']
        },
        'cuda_runtime': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'driver': subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True
        ).strip(),
        'devices': [],
        'model_loaded': False,
    }
    try:
        assert torch.cuda.is_available(), 'CUDA is not available in the serving environment'
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            report['devices'].append(
                {
                    'index': index,
                    'name': props.name,
                    'compute_capability': [props.major, props.minor],
                    'total_memory_bytes': props.total_memory,
                }
            )
        # Exercise kernel loading and GEMM, not only device enumeration.
        x = torch.ones((64, 64), dtype=torch.bfloat16, device='cuda')
        assert (x @ x).float().mean().item() == 64
        torch.cuda.synchronize()
        import vllm
        import vllm._custom_ops as ops

        out = torch.empty_like(x)
        ops.rms_norm(out, x, torch.ones(64, dtype=x.dtype, device=x.device), 1e-6)
        assert torch.allclose(out.float(), torch.ones_like(out).float(), atol=0.01)
        torch.cuda.synchronize()
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

        report['checks'] = [
            'CUDA allocation, kernel execution and bf16 GEMM',
            'vLLM native RMSNorm CUDA kernel',
            'Qwen3.5 architecture import',
        ]
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: CUDA and Qwen serving environment checks; model inference is not yet tested.')


if __name__ == '__main__':
    main()
