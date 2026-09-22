"""Step-4 operator reference: the DSA and fp8-MoE kernels, validated in Triton.

Two tiers:

* ``torch_ref`` -- the operators written out in plain fp32 PyTorch. This is the
  mathematical specification: slow, but the definition of "correct". The reference
  functions are ports of each operator's upstream unit-test reference.
* ``triton_ops`` -- the released Triton/CUDA kernels from ``inference/kernel.py``. Every
  gate runs the Triton kernel and the torch reference on identical inputs and checks they
  agree within a documented tolerance (see ``README.md`` for the envelope per operator).

The gates therefore *validate the Triton implementation against the reference*. A third
party porting these operators to new hardware supplies its own ops module with the same
call signatures as ``triton_ops`` and runs the same gates as an acceptance test, using the
tolerances here as the pass criteria.

Run everything from the repository root:

    python3 -m operator_parity.run_all
"""
