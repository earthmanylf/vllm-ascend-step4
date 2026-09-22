#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""msprof target for the Prefill 16k scene.

Shape (same as previous Triton/Cube case)::

    Q [16384, 8, 192]     TND  BF16
    K/V [317824, 1, 192]  TGD  BF16
    packed [16384, 513]   INT32  dense prefix phys=0..512, valid=8
    counts [16384]        INT32  live=513
    GQA 8/1, D=192, region_size=8, page_size=128

Do not time in Python. Wrap this script with ``msprof op``.
``torch.empty`` + H2D only, so the first NPU kernel is the attention op.
``--iters`` must be > msprof ``warm-up`` + ``launch-count`` (default 5+1).

Keep CANN ``ASCEND_OPP_PATH``. Do not use ``~/tmp_opp`` (dcci binaries, 507015).
CPack vendor (151104B, no dcci)::

    $CPACK/.../packages/vendors/custom_transformer

Smoke (no msprof)::

    cd attention/step4_block_sparse_attention
    # Cube, venv-npu
    source /usr/local/Ascend/cann/set_env.sh
    export ASCEND_CUSTOM_OPP_PATH=$CPACK
    export LD_LIBRARY_PATH=$CPACK/op_api/lib:$LD_LIBRARY_PATH
    unset ASCEND_RT_VISIBLE_DEVICES
    TEST_DEVICE_ID=4 /home/Anthony0331/venv-npu/bin/python -u \\
        examples/bench_triton_vs_cube.py --impl cube --iters 1

    # Triton, catlass-dsl
    source /usr/local/Ascend/cann/set_env.sh
    unset ASCEND_RT_VISIBLE_DEVICES
    TEST_DEVICE_ID=4 /opt/miniforge3/envs/ascend-catlass-dsl/bin/python -u \\
        examples/bench_triton_vs_cube.py --impl triton --iters 1

msprof (after smoke)::

    msprof op --aic-metrics=Default --kernel-name=Step4BlockSparseAttention --kill=on \\
      --application="/home/Anthony0331/venv-npu/bin/python -u examples/bench_triton_vs_cube.py --impl cube" \\
      --output=./msprof_cube_prefill16k

    msprof op --aic-metrics=Default --kernel-name=_sparse_attn_prefill_kernel --kill=on \\
      --application="/opt/miniforge3/envs/ascend-catlass-dsl/bin/python -u examples/bench_triton_vs_cube.py --impl triton" \\
      --output=./msprof_triton_prefill16k
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

# This op's ./triton is a local package, not triton-lang. Drop it before torch
# (venv-npu dynamo imports triton.language) and before the real triton-ascend.
_OP_ROOT = Path(__file__).resolve().parents[1]
_SHADOW = {str(_OP_ROOT), str(_OP_ROOT / "triton")}
sys.path = [p for p in sys.path if p not in _SHADOW and Path(p or ".").resolve() != _OP_ROOT]

import torch
import torch_npu  # noqa: F401

# Previous Prefill 16k case: "16384,8,192;317824,1,192;317824,1,192;16384,513;16384"
QT, N1, D = 16384, 8, 192
KVT, G = 317824, 1
RS, TOPK, PAGE = 8, 513, 128
REGION_ID_MASK = 0x00FFFFFF
VALID_SHIFT = 24


def encode_prefill_i32(phys: int, valid: int) -> int:
    u = (int(phys) & REGION_ID_MASK) | ((int(valid) & 0xFF) << VALID_SHIFT)
    return u - (1 << 32) if u >= (1 << 31) else u


def load_cube_op():
    try:
        from cann_ops_transformer_custom.ops import step4_block_sparse_attention
        return step4_block_sparse_attention
    except ImportError:
        from cann_ops_transformer.ops import step4_block_sparse_attention
        return step4_block_sparse_attention


def load_triton_kernel():
    kernel_path = Path(__file__).resolve().parents[1] / "triton" / "kernel.py"
    spec = importlib.util.spec_from_file_location("step4_triton_kernel", kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {kernel_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._sparse_attn_prefill_kernel


def make_packed(qs: int, kvs: int, g: int) -> tuple[torch.Tensor, torch.Tensor]:
    n_region = max(kvs // RS, 1)
    live = min(TOPK, n_region)
    row = torch.tensor([encode_prefill_i32(p, RS) for p in range(live)], dtype=torch.int32)
    packed = torch.full((g * qs, TOPK), -1, dtype=torch.int32)
    packed[:, :live] = row
    counts = torch.full((g * qs,), live, dtype=torch.int32)
    return packed, counts


def run_cube(op, q, k, v, packed, counts, scale: float):
    return op(
        q, k, v, packed, counts,
        num_heads=N1, num_key_value_heads=G,
        region_size=RS, page_size=PAGE,
        scale_value=scale, return_softmax_lse=True,
    )


def run_triton(kernel, q, k, v, packed, counts, scale: float, block_r: int):
    import triton

    total_q, num_heads, head_dim = q.shape
    heads_per_group = num_heads // G
    topk = int(packed.shape[1])
    block_r = max(1, min(int(block_r), topk))
    out = torch.empty_like(q)
    lse = torch.empty((total_q, num_heads), device=q.device, dtype=torch.float32)
    kernel[(total_q, G)](
        q, k, v, packed, counts, out, lse,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        packed.stride(0),
        out.stride(0), out.stride(1), lse.stride(0),
        scale, total_q,
        topk=topk,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        region_size=RS,
        BLOCK_H=triton.next_power_of_2(heads_per_group),
        BLOCK_D=head_dim,
        BLOCK_R=block_r,
        enable_dynamic_cv_pipeline=False,
        set_workspace_multibuffer=1,
        enable_preload=False,
    )
    return out, lse


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", required=True, choices=("cube", "triton"))
    p.add_argument("--device", type=int, default=int(os.environ.get("TEST_DEVICE_ID", "4")))
    p.add_argument("--iters", type=int, default=8, help="Must exceed msprof warm-up + launch-count")
    p.add_argument("--block-r", type=int, default=16, help="Triton BLOCK_R")
    args = p.parse_args()

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    scale = 1.0 / (D ** 0.5)

    # empty/H2D only: avoid NPU randn (StatelessNormal) stealing msprof's first kernel.
    q = torch.empty(QT, N1, D, dtype=torch.bfloat16, device=device)
    k = torch.empty(KVT, G, D, dtype=torch.bfloat16, device=device)
    v = torch.empty(KVT, G, D, dtype=torch.bfloat16, device=device)
    packed, counts = make_packed(QT, KVT, G)
    packed = packed.contiguous().to(device)
    counts = counts.contiguous().to(device)

    if args.impl == "cube":
        op = load_cube_op()

        def launch():
            return run_cube(op, q, k, v, packed, counts, scale)

        kernel_hint = "Step4BlockSparseAttention"
    else:
        kernel = load_triton_kernel()

        def launch():
            return run_triton(kernel, q, k, v, packed, counts, scale, args.block_r)

        kernel_hint = "_sparse_attn_prefill_kernel"

    print(
        f"impl={args.impl} kernel={kernel_hint} device=npu:{args.device} "
        f"Q=[{QT},{N1},{D}] KV=[{KVT},{G},{D}] packed=[{G * QT},{TOPK}] "
        f"rs={RS} live={TOPK} iters={args.iters}"
        + (f" BLOCK_R={args.block_r}" if args.impl == "triton" else ""),
        flush=True,
    )

    # First call compiles (torch extension / triton JIT). msprof filters by --kernel-name.
    launch()
    torch.npu.synchronize()
    for i in range(args.iters):
        launch()
        torch.npu.synchronize()
        print(f"iter {i} done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
