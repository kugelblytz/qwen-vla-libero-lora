"""Avoid vocabulary projections at ignored targets; retain the causal loss objective."""

from functools import wraps
import torch
from liger_kernel.transformers.model import qwen3_5


def install():
    original = qwen3_5.LigerForCausalLMLoss

    @wraps(original)
    def selected(hidden_states, lm_head_weight, labels, hidden_size, **kwargs):
        # Predicted-token tensors have sequence-shaped consumers; leave that path
        # unchanged. Standard loss-only training/evaluation does not request them.
        if kwargs.get('return_predicted_tokens', False):
            return original(hidden_states, lm_head_weight, labels, hidden_size, **kwargs)
        ignore = kwargs.get('ignore_index', -100)
        shift = kwargs.get('shift_labels')
        if shift is None:
            shift = torch.nn.functional.pad(labels, (0, 1), value=ignore)[..., 1:].contiguous()
        flat_labels = shift.reshape(-1).to(hidden_states.device)
        valid = flat_labels != ignore
        if not valid.any():
            return original(hidden_states, lm_head_weight, labels, hidden_size, **kwargs)
        selected_states = hidden_states.reshape(-1, hidden_size)[valid]
        kwargs = dict(kwargs, shift_labels=flat_labels[valid])
        return original(selected_states, lm_head_weight, None, hidden_size, **kwargs)

    qwen3_5.LigerForCausalLMLoss = selected
    return original, selected
