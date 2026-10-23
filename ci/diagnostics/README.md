# Isolated diagnostics only

These helpers are not imported by the production image or normal launch.
`controlled_mtp_extension.py` and `run_controlled_comparison.py` require an isolated developer service; they switch draft graph dispatch and use pause/resume/collective RPC. Never run them against production or a shared service. They are not benchmark entry points.

The ordered BS16/BS20 FULL/FULL/eager/eager/FULL comparisons passed exact token matching with trace enabled and disabled. This closes the fixed-workload execution-path discrepancy, not arbitrary scheduling/batch-invariance or universal numerical guarantees.

`audit_tailring_capacity.py` replays the installed planner on CPU with explicit Step5 geometry. It does not allocate model/NPU memory and is not a replacement for actual descriptors. The corrected calculation aligns per-1M-request17433 blocks and per-DP23631 blocks. API metrics `num_gpu_blocks=189046` is an eight-DP sum, not a per-DP capacity.

`replay_original_metrics.py` performs offline metric-only replay against a supplied original script and saved iteration/request JSON. It sends no HTTP generation requests; its result is not a fresh run of the original client.
