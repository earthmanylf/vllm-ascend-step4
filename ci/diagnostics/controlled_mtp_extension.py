"""Isolated worker diagnostic extension; never installed in the production image."""

import functools
import json
from pathlib import Path

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_rank


class ControlledMTPExtension:
    def init_worker_extension(self):
        runner = self.model_runner
        runner._controlled_draft_mode = "full"
        runner._controlled_trace_rows = 0
        runner._controlled_trace_enabled = False
        runner._controlled_phase = "initial"
        original_sample = runner.sample

        @functools.wraps(original_sample)
        def sample(hidden_states, input_batch, grammar_output):
            result = original_sample(hidden_states, input_batch, grammar_output)
            if runner._controlled_trace_enabled and get_tensor_model_parallel_rank() == 0:
                output, sampled, rejected = result
                row = {
                    "kind": "sample",
                    "phase": runner._controlled_phase,
                    "req_ids": input_batch.req_ids,
                    "num_reqs": input_batch.num_reqs,
                    "num_tokens": input_batch.num_tokens,
                    "padded_tokens": input_batch.num_tokens_after_padding,
                    "padded_reqs": input_batch.num_reqs_after_padding,
                    "cpu_query": input_batch.query_start_loc_np.tolist(),
                    "device_query": input_batch.query_start_loc.cpu().tolist(),
                    "seq_lens": input_batch.seq_lens.cpu().tolist(),
                    "computed": input_batch.num_computed_tokens_np.tolist(),
                    "scheduled": input_batch.num_scheduled_tokens.tolist(),
                    "is_prefilling": input_batch.is_prefilling_np.tolist(),
                    "logits_indices": input_batch.logits_indices.cpu().tolist(),
                    "cu_num_logits": input_batch.cu_num_logits_np.tolist(),
                    "num_sampled": sampled.cpu().tolist(),
                    "num_rejected": rejected.cpu().tolist(),
                    "sampled_tokens": output.sampled_token_ids.cpu().tolist(),
                }
                self._controlled_write(row)
            return result

        runner.sample = sample
        original_init = runner.initialize_kv_cache

        @functools.wraps(original_init)
        def initialize_kv_cache(*args, **kwargs):
            result = original_init(*args, **kwargs)
            if kwargs.get("is_profiling", False):
                return result
            spec = runner.speculator
            if spec is None:
                raise RuntimeError("MTP diagnostic extension requires a draft model")
            manager = spec.cudagraph_manager
            original_dispatch = manager.dispatch

            @functools.wraps(original_dispatch)
            def dispatch(*args, **kwargs):
                desc = original_dispatch(*args, **kwargs)
                if runner._controlled_draft_mode == "eager":
                    from dataclasses import replace

                    desc = replace(desc, cg_mode=CUDAGraphMode.NONE)
                if runner._controlled_trace_enabled and get_tensor_model_parallel_rank() == 0:
                    self._controlled_write(
                        {
                            "kind": "dispatch",
                            "phase": runner._controlled_phase,
                            "descriptor": str(desc),
                            "dispatch_inputs": str(args),
                        }
                    )
                return desc

            manager.dispatch = dispatch
            original_build = spec._build_attn_metadata

            @functools.wraps(original_build)
            def build(*args, **kwargs):
                metadata = original_build(*args, **kwargs)
                if runner._controlled_trace_enabled and get_tensor_model_parallel_rank() == 0:
                    self._controlled_write(
                        {
                            "kind": "metadata",
                            "phase": runner._controlled_phase,
                            "layers": {
                                name: {
                                    "query": value.actual_seq_lengths_q,
                                    "kv": value.seq_lens_list,
                                    "slots": value.slot_mapping.cpu().tolist(),
                                }
                                for name, value in metadata.items()
                                if value is not None and hasattr(value, "actual_seq_lengths_q")
                            },
                        }
                    )
                return metadata

            spec._build_attn_metadata = build
            original_propose = spec.propose

            @functools.wraps(original_propose)
            def propose(input_batch, *args, **kwargs):
                result = original_propose(input_batch, *args, **kwargs)
                if runner._controlled_trace_enabled and get_tensor_model_parallel_rank() == 0:
                    rows = input_batch.num_reqs
                    metadata = getattr(spec, "_multi_module_attn_metadata", None)
                    self._controlled_write(
                        {
                            "kind": "draft",
                            "phase": runner._controlled_phase,
                            "mode": runner._controlled_draft_mode,
                            "req_ids": input_batch.req_ids,
                            "num_tokens": input_batch.num_tokens,
                            "padded_tokens": input_batch.num_tokens_after_padding,
                            "draft_tokens": result.cpu().tolist(),
                            "draft_query": spec.input_buffers.query_start_loc[: rows + 1].cpu().tolist(),
                            "draft_seq_lens": spec.input_buffers.seq_lens[:rows].cpu().tolist(),
                            "metadata_retained": metadata is not None,
                        }
                    )
                return result

            spec.propose = propose
            return result

        runner.initialize_kv_cache = initialize_kv_cache

    def controlled_mtp_configure(self, mode="full", phase="diagnostic", trace="true"):
        if mode not in ("full", "eager"):
            raise ValueError(mode)
        if not phase.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Invalid phase")
        runner = self.model_runner
        runner._controlled_draft_mode = mode
        runner._controlled_phase = phase
        runner._controlled_trace_enabled = trace == "true"
        runner._controlled_trace_rows = 0
        return {
            "rank": self.rank,
            "mode": mode,
            "phase": phase,
            "target_graph_mode": str(runner.cudagraph_manager.cudagraph_mode),
        }

    def _controlled_write(self, row):
        runner = self.model_runner
        if runner._controlled_trace_rows >= 400:
            return
        phase = runner._controlled_phase
        path = Path("/run-cluster") / f"trace-{phase}-rank{self.rank}.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        runner._controlled_trace_rows += 1
