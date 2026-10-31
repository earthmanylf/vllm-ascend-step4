# CANN-only A5 / Step5 unified Dockerfile

The native recipe is `ci/docker/Dockerfile`. It combines the common A5 setup
from `Dockerfile.a5`, the matching openEuler package setup from
`Dockerfile.a5.openEuler`, and the existing Step5 CI source/version/custom-op
flow. It does **not** use a completed vllm-ascend image as its base.

Default base:

```text
hub.i.basemind.com/stepcast/stepcast:cann-9.2.0_20261022115200-950-openeuler24.0
```

## Stage layout

| Stage | Contents |
| --- | --- |
| `cann-base` | The supplied CANN image only |
| `a5-system` | A5/openEuler compiler and runtime system dependencies |
| `builder` | Python 3.12 venv, torch/NPU, fixed vLLM source, native Ascend wheel, FA3/Step5 ops, Mooncake, Rust and optional A5 packages |
| `runtime` | Inherits `a5-system`, then copies installed venv, editable vLLM source, selected vendor packages and source identities |

The runtime does not inherit `builder`, its `.git`, native build tree, Cargo
cache, or source archives. Stage-aware budgeting currently counts 16 added
filesystem instructions on the deepest builder path and 8 on the final runtime
path, excluding the actual CANN base's existing layers. These are recipe
budgets, not a claim that the final image has already been built.

## Preserved Step5 contract

- Fixed vLLM and vllm-ascend commits and 0.30-series versions are prepared before
  build and passed together; the runtime receipt validates the prepared version,
  not a historical hardcoded 0.29 version.
- torch 2.10 CPU, torch-npu 2.10.0.post4 and the existing triton-ascend build are
  kept for the current Step5 profile rather than switching to the original A5
  file's floating vLLM release/daily dependencies.
- Production FA3 and Step5 custom ops are built from the locked source archives.
  Optional union vendors remain opt-in and do not replace production GBSA by
  default.
- CANN CLI Python paths and the toolkit alias are discovered from the new base;
  the old `/usr/local/python3.12.13` interpreter is no longer assumed.
- No runtime source overlay or inherited vllm-ascend wheel is used.

## Inputs and validation boundary

A CANN-only base supplies the toolkit, **not** torch, vLLM, flash-attn-npu,
production Step5 ops, or vendored C++ source trees. Consequently,
`--reuse-base-assets` is retired for this recipe. The old 4.8GB CANN payload and
unused uv wheel are not required or staged.

The native source archives `ascend-catlass.tar.gz`, `custom-assets.tar.gz`, and
`ops-git-metadata.tar.gz` are still required and checksum-locked. The current
local cache contains LFS pointers for them: rehydrate the inputs before an
actual build. Copied `ci/third-party/prebuilt` union packages do not replace
production FA3/Step5 source archives.

The supplied CANN image must be inspected after download for Python 3.12,
CANN 9.2.0, developer HAL stub, system package repositories and any required
NNAL/ATB components. Host-only checks do not establish those facts. No image
pull, Docker build or NPU regression was run during this Dockerfile fusion.

```bash
cd /data/chensiyu/source/vllm-ascend
./ci/build.sh --check-only
# After the user confirms that the base and source inputs are available:
./ci/build.sh --full --source-ref <fixed-commit> \
  --base-image hub.i.basemind.com/stepcast/stepcast:cann-9.2.0_20261022115200-950-openeuler24.0 \
  --vllm-commit b7042c5887ba8da9c1c0fa0b75d5579893ec5059 --keep
```

The builder inspects the already-local base and uses its digest/config ID; it
uses `--pull=false` and does not silently fetch another base. An explicit
`--python-overlay` remains available only for a previously validated native
image with unchanged csrc. It is not the default CANN-only build path.
