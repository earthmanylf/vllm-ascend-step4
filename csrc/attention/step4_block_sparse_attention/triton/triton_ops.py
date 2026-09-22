"""Thin wrappers around the released Triton kernels in ``inference/kernel.py``.

These are the reference implementation the gates validate: each wrapper exposes one
Step-4 operator with a stable signature. A third party can drop in an equivalent module
for their own kernels and run the same gates. All symbols come from the consolidated
``inference.kernel`` module.
"""

from __future__ import annotations


def triton_sparse_attention_prefill(*args, **kwargs):
    """Sparse attention over the selected regions, prefill path (stage 5)."""
    from inference.kernel import sparse_attention_prefill

    return sparse_attention_prefill(*args, **kwargs)



