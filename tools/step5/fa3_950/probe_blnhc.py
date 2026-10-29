"""Actual upstream planner/allocator -> native strided writer/FA3 FULL replay.

Not full-model MTP acceptance. Uses the installed source from the built image,
without substituting an allocator or mocking a device operation.
This is a diagnostic boundary test, not final-image/model acceptance.
"""

import csv
import hashlib
import importlib.metadata as md
import json
import math
import os
import pathlib
import subprocess
import sys
from contextlib import ExitStack
from types import SimpleNamespace as N

import torch
import torch_npu
from check_library import EXPECTED_SYMBOL
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups, get_kv_cache_groups
from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import allocate_kv_cache
from vllm_ascend.attention.flash_attention_v3 import AscendFlashAttentionImpl, AscendFlashAttentionMetadataBuilder
from vllm_ascend.core.step5_kv_cache import step5_blnhc_layer_views, validate_step5_blnhc_descriptors

# Match the real Step5 model import order; ops registration precedes device_op.
from vllm_ascend.device.device_op import DeviceOperator

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
config = N(
    use_v2_model_runner=True,
    model_config=N(
        hf_text_config=N(model_type="step5"), dtype=torch.bfloat16, max_model_len=512, runner_type="generate"
    ),
    scheduler_config=N(disable_hybrid_kv_cache_manager=False, max_num_seqs=2),
    parallel_config=N(
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        tensor_parallel_size=4,
    ),
    attention_config=N(hisparse_config=None),
    cache_config=N(
        kv_cache_layout="BLNHC",
        get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLNHC,
        num_gpu_blocks_override=None,
        prefix_cache_retention_interval=0,
    ),
    additional_config={},
    kv_transfer_config=None,
    speculative_config=N(),
    num_prefill_lookahead_tokens=0,
    compilation_config=N(cudagraph_capture_sizes=[]),
)
geometry = dict(block_size=128, num_kv_heads=1, head_size=192, dtype=torch.bfloat16, block_stride_alignment=384)
specs = {
    "dense": FullAttentionSpec(**geometry),
    "summary": FullAttentionSpec(
        **dict(geometry, head_size=256, head_size_v=0, dtype=torch.uint8, block_stride_alignment=256),
        tokens_per_state=8,
    ),
    "ring": CircularBufferSpec(**dict(geometry, head_size=384, head_size_v=0, block_stride_alignment=1024)),
}
groups = get_kv_cache_groups(config, specs)
planned = get_kv_cache_config_from_groups(config, groups, 8 * 1024 * 1024)
checked = validate_step5_blnhc_descriptors(planned, KVCacheLayout.BLNHC)
canonical = allocate_kv_cache(planned, torch.device("npu:0"), KVCacheLayout.BLNHC, [128] * len(groups))
caches = step5_blnhc_layer_views(canonical, checked)
assert specs["summary"].page_size_padded is None
k, v = caches["dense"]
blocks = k.shape[0]
assert blocks >= 12
assert (
    len({t.untyped_storage().data_ptr() for c in caches.values() for t in (c if isinstance(c, tuple) else (c,))}) == 1
)
assert k.stride(0) > 128 * 192 and v.stride(0) == k.stride(0)
torch.manual_seed(721)
device = torch.device("npu:0")
heads = 16
dim = 192
k_cpu = torch.zeros(blocks, 128, 1, dim, dtype=torch.bfloat16)
v_cpu = torch.zeros_like(k_cpu)
initial_pages = [3, 5, blocks - 1, 7, 2, blocks - 2]
values_k = torch.randn(768, 1, dim, device=device, dtype=torch.bfloat16)
values_v = torch.randn_like(values_k)
initial_slots = torch.tensor([p * 128 + t for p in initial_pages for t in range(128)], device=device, dtype=torch.int32)
DeviceOperator.reshape_and_cache(values_k, values_v, k, v, initial_slots)
torch_npu.npu.synchronize()
for i, p in enumerate(initial_pages):
    k_cpu[p] = values_k[i * 128 : (i + 1) * 128].cpu()
    v_cpu[p] = values_v[i * 128 : (i + 1) * 128].cpu()
q_cpu = torch.randn(8, heads, dim, dtype=torch.bfloat16)
q = q_cpu.to(device)
output = torch.empty_like(q)
updates_k = torch.zeros(8, 1, dim, device=device, dtype=torch.bfloat16)
updates_v = torch.zeros_like(updates_k)
impl = AscendFlashAttentionImpl.__new__(AscendFlashAttentionImpl)
impl.num_heads, impl.num_kv_heads, impl.head_size = heads, 1, dim
impl.scale = 1 / math.sqrt(dim)
impl.logits_soft_cap = 0.0
impl.window_size = (127, 0)
builder = AscendFlashAttentionMetadataBuilder(specs["dense"], ["dense"], config, device)
builder.bind_fa3_kv_cache({"dense": N(impl=impl, kv_cache=(k, v))})


def common(lengths, tables, real):
    ends = [0, 4, 8] if real == 2 else [0, 4, 4]
    slots = []
    for r in range(real):
        for p in range(lengths[r] - 4, lengths[r]):
            slots.append(tables[r][p // 128] * 128 + p % 128)
    slots += [777] * (8 - len(slots))
    return N(
        num_reqs=2,
        num_actual_tokens=8,
        num_input_tokens=8,
        max_query_len=4,
        query_start_loc=torch.tensor(ends, device=device, dtype=torch.int32),
        seq_lens=torch.tensor(lengths, device=device, dtype=torch.int32),
        block_table_tensor=torch.tensor(tables, device=device, dtype=torch.int32),
        slot_mapping=torch.tensor(slots, device=device, dtype=torch.int32),
        causal=True,
        attn_state=None,
    )


initial = common([256, 384], [[3, 5, blocks - 1], [7, 2, blocks - 2]], 2)
captured = builder.build_for_cudagraph_capture(initial, num_actual_reqs=2)


def forward():
    DeviceOperator.reshape_and_cache(updates_k, updates_v, k, v, captured.slot_mapping)
    return impl.forward_impl(q, None, None, (k, v), captured, output)


forward()
torch_npu.npu.synchronize()
graph = torch_npu.npu.NPUGraph()
with torch_npu.npu.graph(graph):
    forward()
proof = []
trace_requested = os.environ.get("PROOF_CAPTURE_TRACE") == "1"
profile_scope = ExitStack()
profiler = None
if trace_requested:
    profiler = profile_scope.enter_context(
        torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out / "trace")),
        )
    )
for real, lengths, tables in [
    (2, [256, 384], [[3, 5, blocks - 1], [7, 2, blocks - 2]]),
    (1, [129, 0], [[3, 5, blocks - 1], [0, 0, 0]]),
    (2, [257, 384], [[3, 5, blocks - 1], [7, 2, blocks - 2]]),
]:
    source = common(lengths, tables, real)
    metadata = builder.build(0, source, num_actual_reqs=real, full_graph_mode=True)
    assert metadata.slot_mapping.data_ptr() == captured.slot_mapping.data_ptr()
    assert metadata.block_tables.data_ptr() == captured.block_tables.data_ptr()
    new_k = torch.randn(8, 1, dim, dtype=torch.bfloat16)
    new_v = torch.randn_like(new_k)
    updates_k.copy_(new_k)
    updates_v.copy_(new_v)
    actual_slots = metadata.slot_mapping.cpu().tolist()
    for i, slot in enumerate(actual_slots):
        if slot >= 0:
            k_cpu[slot // 128, slot % 128] = new_k[i]
            v_cpu[slot // 128, slot % 128] = new_v[i]
    graph.replay()
    torch_npu.npu.synchronize()
    expected = []
    for r in range(real):
        ks = k_cpu[torch.tensor(tables[r])].flatten(0, 1)[: lengths[r]].repeat_interleave(heads, dim=1).float()
        vs = v_cpu[torch.tensor(tables[r])].flatten(0, 1)[: lengths[r]].repeat_interleave(heads, dim=1).float()
        for t in range(4):
            pos = lengths[r] - 4 + t
            start = max(0, pos - 127)
            scores = torch.einsum("hd,shd->hs", q_cpu[r * 4 + t].float(), ks[start : pos + 1]) * impl.scale
            expected.append(torch.einsum("hs,shd->hd", scores.softmax(-1), vs[start : pos + 1]))
    expected = torch.stack(expected)
    actual = output[: real * 4].cpu().float()
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    proof.append(
        {"real_reqs": real, "max_abs_error": float((actual - expected).abs().max()), "writer_slots": actual_slots}
    )
    if profiler is not None:
        profiler.step()
profile_scope.close()
libraries = []
symbols = set()
matches = []
if trace_requested:
    distribution = md.distribution("flash-attn-npu")
    for entry in distribution.files:
        if str(entry).endswith(".so"):
            library = pathlib.Path(distribution.locate_file(entry))
            libraries.append({"path": str(library), "sha256": hashlib.sha256(library.read_bytes()).hexdigest()})
            for symbol in subprocess.check_output(["strings", str(library)], text=True).splitlines():
                if symbol == EXPECTED_SYMBOL:
                    symbols.add(symbol)
    assert symbols, "The installed image lacks the complete requested two-stride kernel"
    for trace in (out / "trace").rglob("*.csv"):
        if "kernel" not in trace.name.lower():
            continue
        for row in csv.DictReader(trace.read_text(errors="replace").splitlines()):
            for value in row.values():
                if value in symbols:
                    matches.append(
                        {
                            "file": str(trace),
                            "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                            "symbol": value,
                            "details": row,
                        }
                    )
    assert len(matches) >= 3, "Three fresh exact-kernel graph replay events were not observed"
report = {
    "layout": "BLNHC",
    "source_files": {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            pathlib.Path(__import__("vllm_ascend.core.step5_kv_cache", fromlist=["__file__"]).__file__),
            pathlib.Path(__import__("vllm_ascend.attention.flash_attention_v3", fromlist=["__file__"]).__file__),
        ]
    },
    "summary_page_bytes": specs["summary"].page_size_bytes,
    "image_id": os.environ["PROOF_IMAGE_ID"],
    "test_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    "main_planner_and_original_allocator": True,
    "single_unique_backing": True,
    "blocks": blocks,
    "kv_stride_elements": [k.stride(0), v.stride(0)],
    "head_size": dim,
    "kv_heads": 1,
    "full_graph_requests": [2, 1, 2],
    "actual_device_writer_and_fa3_reader": True,
    "numerical": proof,
    "trace_requested": trace_requested,
    "libraries": libraries,
    "exact_kernel_device_events": matches,
    "full_model_mtp": "not proved by this boundary test",
}
(out / "result.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
print("MAIN_ALLOCATOR_FA3_FULL_GRAPH_BOUNDARY_PASSED")
