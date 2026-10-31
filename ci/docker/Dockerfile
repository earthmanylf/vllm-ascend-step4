#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

# A5/openEuler bootstrap from Dockerfile.a5 and Dockerfile.a5.openEuler,
# fused with the Step5 source/version/FA3/custom-op CI recipe.
# No prebuilt vllm-ascend image and no second CANN payload extraction.
ARG BASE_IMAGE=hub.i.basemind.com/stepcast/stepcast:cann-9.2.0_20261022115200-950-openeuler24.0
FROM ${BASE_IMAGE} AS cann-base

FROM cann-base AS a5-system
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# Keep the A5 toolchain available at runtime too: Step5 has torch-extension JIT.
RUN set -e; yum install -y git wget curl protobuf protobuf-devel protobuf-compiler \
    gcc gcc-c++ make cmake numactl numactl-devel libibverbs-devel hiredis-devel \
    clang patch findutils tar gzip which libxcb && yum clean all && rm -rf /var/cache/yum

FROM a5-system AS builder
ARG BASE_IMAGE
ARG PYTHON_BIN=python3
# Internal Artifactory is the primary index: pypi-public is a virtual repo in
# front of PyPI/aliyun and pytorch-public proxies download.pytorch.org. The
# public PyPI host is effectively unreachable from the build node.
ARG PIP_INDEX_URL=https://artifactory.stepfun-inc.com/artifactory/api/pypi/pypi-public/simple
# pypi-public is a cached virtual index and can lag upstream (e.g. it stopped at
# mistral-common 1.9.1 while vLLM requires >=1.11.6); pypi-org is fresh and also
# serves artifact-relative URLs, so downloads stay inside Artifactory.
ARG PIP_EXTRA_INDEX_URL=https://artifactory.stepfun-inc.com/artifactory/api/pypi/pypi-org/simple
ARG PYTORCH_INDEX_URL=https://artifactory.stepfun-inc.com/artifactory/api/pypi/pytorch-cpu/simple
ARG ASCEND_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi
ARG OPENCV_VERSION=4.13.0.90
ARG VLLM_REPO=https://gitlab.basemind.com/sys/stepcast/vllm.git
ARG VLLM_COMMIT
ARG VLLM_VERSION=
ARG VLLM_BASE_VERSION=0.30.0rc1
ARG VLLM_ASCEND_VERSION
ARG VLLM_ASCEND_COMMIT
ARG VLLM_ASCEND_BASE_VERSION=0.30.0rc1
ARG GITLAB_ACCESS_TOKEN=
ARG SOC_VERSION=ascend950dt_9582
ARG MAX_JOBS=16
ARG CARGO_BUILD_JOBS=16
ARG WITH_UNION_OPS=0
ARG BUILD_RUST_FRONTEND=1
ARG INSTALL_A5_OPTIONAL_OPS=1
ARG GIT_PROXY=
ENV TORCH_DEVICE_BACKEND_AUTOLOAD=0 SOC_VERSION=${SOC_VERSION} \
    MAX_JOBS=${MAX_JOBS} CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS} OMP_NUM_THREADS=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 PIP_PROGRESS_BAR=off PIP_DEFAULT_TIMEOUT=120
WORKDIR /opt/upgrade
COPY normalize-cann92.py cann92-env.sh cann92-build-env.sh cann92-entrypoint.sh \
    apply_dependency_profile.py probe_environment.py install_ops.py patch_flash_hcc.py /opt/upgrade/
# Python paths differ between CANN distributions. Resolve and check rather than
# inheriting /usr/local/python3.12.13 from a vllm-ascend release image.
RUN set -e; selected=; \
    for p in "$PYTHON_BIN" python3.12 /usr/local/python3.12.13/bin/python3.12 /usr/local/python3.12.15/bin/python3.12; do \
        if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; assert sys.version_info[:2] == (3,12)' 2>/dev/null; then selected="$p"; break; fi; \
    done; \
    test -n "$selected" || { echo 'CANN base must supply Python 3.12; inspect the downloaded image or set PYTHON_BIN' >&2; exit 1; }; \
    "$selected" -m venv /opt/upgrade/.venv; \
    /opt/upgrade/.venv/bin/python /opt/upgrade/normalize-cann92.py
ENV PATH=/opt/upgrade/.venv/bin:/opt/upgrade/cann-cli/bin:/root/.cargo/bin:${PATH}
# Fresh CANN-only base: explicitly install the audited torch/NPU stack.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked set -e; source /opt/upgrade/cann92-build-env.sh; \
    python3 -m pip install --upgrade pip 'setuptools>=77.0.3,<81' wheel; \
    python3 -m pip install --extra-index-url "$PYTORCH_INDEX_URL" \
        'torch==2.10.0+cpu' 'torchvision==0.25.0+cpu' 'torchaudio==2.10.0+cpu' 'numpy<2'; \
    python3 -m pip install --extra-index-url "$ASCEND_INDEX_URL" 'torch-npu==2.10.0.post4'

# Sources are immutable inputs. Credentials and .git never enter runtime stage.
RUN set -euo pipefail; test -n "$VLLM_COMMIT"; test -n "$VLLM_VERSION"; \
    mkdir -p /opt/upgrade/src /opt/upgrade/evidence /opt/upgrade/wheels; \
    export GIT_TERMINAL_PROMPT=0; url="$VLLM_REPO"; \
    if [[ -n "$GITLAB_ACCESS_TOKEN" && "$url" == https://* ]]; then url="https://oauth2:${GITLAB_ACCESS_TOKEN}@${url#https://}"; fi; \
    git clone "$url" /opt/upgrade/src/vllm; \
    git -C /opt/upgrade/src/vllm checkout --detach "$VLLM_COMMIT"; \
    actual="$(git -C /opt/upgrade/src/vllm rev-parse HEAD)"; test "$actual" = "$VLLM_COMMIT"; \
    case "$VLLM_VERSION" in *"$actual"*) ;; *) echo 'vllm version does not identify cloned commit' >&2; exit 1;; esac; \
    printf '%s\n' "$actual" > /opt/upgrade/src/vllm-commit.txt; \
    printf '%s\n' "$VLLM_VERSION" > /opt/upgrade/src/vllm-version.txt; \
    python3 -c 'import json,os,pathlib; p=pathlib.Path("/opt/upgrade/src/vllm/SOURCE-IDENTITY.json"); p.write_text(json.dumps({"commit":os.environ["VLLM_COMMIT"],"source_version":os.environ["VLLM_VERSION"],"base_version":os.environ["VLLM_BASE_VERSION"]},indent=2)+"\n")'; \
    rm -rf /opt/upgrade/src/vllm/.git
COPY vllm-ascend.tar.gz ascend-catlass.tar.gz custom-assets.tar.gz ops-git-metadata.tar.gz /opt/upgrade/input/
RUN set -euo pipefail; test -n "$VLLM_ASCEND_COMMIT"; test -n "$VLLM_ASCEND_VERSION"; \
    mkdir -p /opt/upgrade/src/vllm-ascend; \
    tar --no-same-owner -xf /opt/upgrade/input/vllm-ascend.tar.gz -C /opt/upgrade/src/vllm-ascend; \
    mkdir -p /opt/upgrade/src/vllm-ascend/csrc/third_party/catlass; \
    tar --no-same-owner -xzf /opt/upgrade/input/ascend-catlass.tar.gz -C /opt/upgrade/src/vllm-ascend/csrc/third_party/catlass; \
    tar --no-same-owner -xzf /opt/upgrade/input/custom-assets.tar.gz -C /opt/upgrade; \
    tar --no-same-owner -xzf /opt/upgrade/input/ops-git-metadata.tar.gz -C /opt/upgrade; \
    test -d /opt/upgrade/src/vllm-ascend/csrc/third_party/catlass/include; \
    for d in json makeself pkg; do \
        test -d "/opt/upgrade/source-assets/ascend-thirdparty/$d"; \
        ln -sfn "/opt/upgrade/source-assets/ascend-thirdparty/$d" "/opt/upgrade/src/vllm-ascend/csrc/third_party/$d"; \
    done; \
    rm -rf /opt/upgrade/input; python3 /opt/upgrade/apply_dependency_profile.py

COPY triton_ascend-3.6.0+dev20261013182112-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl /opt/upgrade/wheel-input/
# A5 base features plus the existing Step5 dependency profile. Do not let a
# source package silently upgrade torch/CANN away from the audited stack.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked set -e; source /opt/upgrade/cann92-build-env.sh; \
    python3 -m pip uninstall -y triton triton-ascend || true; \
    wheel="/opt/upgrade/wheel-input/triton_ascend-3.6.0+dev20261013182112-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"; \
    printf 'd46b9b80a3e0cb21e1965410acacf800b47bd16a150e995f06194b01aaf59c03  %s\n' "$wheel" | sha256sum -c -; \
    python3 -m pip install --no-deps "$wheel"; \
    python3 -m pip install -c /opt/upgrade/constraints-tailring.txt \
        -r /opt/upgrade/src/vllm/requirements/common.txt \
        'transformers==5.19.0' 'modelscope<1.38' 'ray>=2.47.1,<=2.48.0' \
        'protobuf>3.20.0' concurrent-log-handler 'build==1.6.1' 'expecttest==0.3.0' 'pyelftools==0.32'; \
    python3 -m pip install -c /opt/upgrade/constraints-tailring.txt \
        -r /opt/upgrade/src/vllm-ascend/requirements.txt --extra-index-url "$ASCEND_INDEX_URL" --extra-index-url "$PYTORCH_INDEX_URL"; \
    python3 -c 'import json,os,pathlib,subprocess,sys,tomllib; root=pathlib.Path("/opt/upgrade/src"); req=sorted({r for repo in ("vllm","vllm-ascend") for r in tomllib.loads((root/repo/"pyproject.toml").read_text())["build-system"]["requires"]}); pathlib.Path("/opt/upgrade/evidence/build-requirements.json").write_text(json.dumps(req,indent=2)); subprocess.run([sys.executable,"-m","pip","install","-c","/opt/upgrade/constraints-tailring.txt","--extra-index-url",os.environ["ASCEND_INDEX_URL"],"--extra-index-url",os.environ["PYTORCH_INDEX_URL"],*req],check=True)'; \
    # torch depends on the upstream `triton` distribution, whose wheel ships a
    # CUDA libtriton.so and therefore overwrites the Ascend build when it lands
    # after triton-ascend. Reinstall triton-ascend last so the Ascend backend
    # (triton/_C/libtriton.so + triton/backends/ascend) is the effective one.
    python3 -m pip install --force-reinstall --no-deps "$wheel"; \
    rm -rf /opt/upgrade/wheel-input; \
    python3 -m pip install --no-deps "opencv-python-headless==$OPENCV_VERSION"; \
    python3 -c 'import numpy, cv2; assert numpy.__version__ == "1.26.4", numpy.__version__; assert cv2.__version__.startswith("4.13."), cv2.__version__; import numpy as _np; assert cv2.imencode(".png", _np.zeros((8, 8, 3), _np.uint8))[0]'; \
    python3 -c 'import triton, triton.language as tl, triton.backends; assert hasattr(tl, "extra"), "triton.language.extra missing: upstream triton overwrote triton-ascend"; print("triton", triton.__version__, sorted(triton.backends.backends))'

# Mooncake installation follows the original A5/openEuler recipe.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/var/cache/remote-wheels,sharing=locked \
    set -e; source /opt/upgrade/cann92-build-env.sh; \
    wheel='mooncake_transfer_engine-0.3.13-cp312-cp312-manylinux_2_35_aarch64.whl'; \
    cached="/var/cache/remote-wheels/$wheel"; \
    test -s "$cached" || curl -fL --retry 5 --retry-all-errors --retry-delay 3 -C - \
        "https://mindie.obs.cn-north-4.myhuaweicloud.com/buildtools/docker_images_tools/Mooncake/v0.3.14.rc1/$wheel" -o "$cached"; \
    python3 -m pip install --no-deps "$cached"
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/root/.rustup,sharing=locked \
    --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    set -e; source /opt/upgrade/cann92-build-env.sh; \
    if [ "$BUILD_RUST_FRONTEND" = 1 ]; then \
        if ! command -v rustup >/dev/null 2>&1; then \
            curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain none; \
        fi; \
        toolchain="$(python3 -c 'import pathlib,tomllib; print(tomllib.loads(pathlib.Path("/opt/upgrade/src/vllm/rust-toolchain.toml").read_text())["toolchain"]["channel"])')"; \
        rustup toolchain install "$toolchain"; \
    fi; \
    SETUPTOOLS_SCM_PRETEND_VERSION="$VLLM_VERSION" VLLM_TARGET_DEVICE=empty \
        python3 -m pip install -e /opt/upgrade/src/vllm --no-build-isolation --no-deps; \
    if [ -d /opt/upgrade/src/vllm/stepcast_plugin ]; then \
        python3 -m pip install --no-deps /opt/upgrade/src/vllm/stepcast_plugin; \
    fi; \
    if [ "$BUILD_RUST_FRONTEND" = 1 ]; then \
        cd /opt/upgrade/src/vllm; python3 -m pip install -r requirements/build/rust.txt; \
        : "${VLLM_VERSION:?VLLM_VERSION is required for the rust frontend build}"; \
        SETUPTOOLS_SCM_PRETEND_VERSION="$VLLM_VERSION" \
        SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_RUST_FRONTEND_BUILD="$VLLM_VERSION" \
        VCS_VERSIONING_PRETEND_VERSION_FOR_VLLM_RUST_FRONTEND_BUILD="$VLLM_VERSION" \
        VLLM_RS_BUILD_VERSION="$VLLM_VERSION" \
            PROTOC_INCLUDE=/usr/include bash tools/build_rust.sh; \
    fi
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked set -e; source /opt/upgrade/cann92-build-env.sh; \
    cd /opt/upgrade/src/vllm-ascend; \
    COMPILE_CUSTOM_KERNELS=1 SETUPTOOLS_SCM_PRETEND_VERSION="$VLLM_ASCEND_VERSION" \
        python3 -m pip wheel . --no-build-isolation --no-deps -w /opt/upgrade/wheels; \
    python3 -m pip install --no-deps /opt/upgrade/wheels/vllm_ascend-*.whl
# FA3/production Step5 ops must be built; a CANN-only base has neither.
# native.patch widens the v3 FAInfer template to 16 arguments; v4's untouched
# mha_bwd.cpp still calls the 14-argument form, so build the v3 950 backend
# only. FA3 uses flash_attn_npu_3_950; v4 is not part of this service.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked set -e; source /opt/upgrade/cann92-build-env.sh; \
    # The paged block-stride ABI is a compiled property of the FA3 operator, not
    # something Python can add later: vllm_ascend refuses to run without it.
    # This patch must go first: it rewrites setup.py (including the hcc
    # libstdc++ include lookup), and it verifies the pristine source hashes.
    python3 /opt/upgrade/src/vllm-ascend/tools/step5/fa3_950/apply.py \
        --source /opt/upgrade/source-assets/flash-attention-npu; \
    python3 /opt/upgrade/patch_flash_hcc.py; \
    cd /opt/upgrade/source-assets/flash-attention-npu; \
    FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=950 python3 setup.py bdist_wheel; \
    python3 -m pip install --no-deps dist/flash_attn_npu-*.whl; \
    abi_so="$(ls /opt/upgrade/.venv/lib/python3.12/site-packages/flash_attn_npu_3_950*.so)"; \
    python3 /opt/upgrade/src/vllm-ascend/tools/step5/fa3_950/check_library.py "$abi_so" \
        > /opt/upgrade/evidence/fa3-abi.json; \
    python3 -c 'import json; d=json.load(open("/opt/upgrade/evidence/fa3-abi.json")); assert d["abi_version"] == 1, d; print("FA3 block-stride ABI", d["abi_version"], "verified on", d["library"])'; \
    cd /opt/upgrade/source-assets/ops-transformer; \
    bash build.sh --pkg --soc=ascend950 --vendor_name=step5_ops \
        --ops=generic_block_sparse_attention,generic_block_sparse_attention_metadata,indexer_region_score \
        --cann_3rd_lib_path=/opt/upgrade/source-assets/ops-thirdparty -j"$MAX_JOBS"; \
    bash build.sh --torch_extension; python3 /opt/upgrade/install_ops.py build_out
# Original A5 optional FLA/cannbot packages remain explicit, not inherited.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/var/cache/remote-wheels,sharing=locked \
    set -e; source /opt/upgrade/cann92-build-env.sh; \
    if [ "$INSTALL_A5_OPTIONAL_OPS" = 1 ]; then \
        wheel='flash_linear_attention_npu_a5-26.9.1+dev5cc6990-py3-none-manylinux_2_34_aarch64.whl'; \
        cached="/var/cache/remote-wheels/$wheel"; \
        test -s "$cached" || curl -fL --retry 5 --retry-all-errors --retry-delay 3 -C - \
            "${GIT_PROXY}https://github.com/flashserve/flash-linear-attention-npu/releases/download/v26.9.1-20261020-020004/$wheel" -o "$cached"; \
        printf '0aa5c9191d536136a6c40a609d25901d7f632cab10d784f10de0c517d07c9c51  %s\n' "$cached" | sha256sum -c -; \
        python3 -m pip install --no-deps "$cached"; \
        for wheel in cannbotdsl-0.6.0+g91e7c8f-cp312-cp312-manylinux_2_28_aarch64.whl cannbot_arena_net_ops-0.1.0-cp312-cp312-linux_aarch64.whl; do \
            cached="/var/cache/remote-wheels/$wheel"; \
            test -s "$cached" || curl -fL --retry 5 --retry-all-errors --retry-delay 3 -C - \
                "https://raw.githubusercontent.com/GDzhu01/cannbot-wheels/main/$wheel" -o "$cached"; \
            python3 -m pip install --no-deps "$cached"; \
        done; \
    fi
COPY third-party/prebuilt/ /opt/upgrade/third-party-prebuilt/
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked set -e; source /opt/upgrade/cann92-build-env.sh; \
    if [ "$WITH_UNION_OPS" = 1 ]; then \
        for v in union_test_transformer custom_nn_nn; do \
            test -d "/opt/upgrade/third-party-prebuilt/$v/op_api/lib"; \
            cp -a "/opt/upgrade/third-party-prebuilt/$v" "$ASCEND_OPP_PATH/vendors/$v"; \
        done; \
        for wheel in /opt/upgrade/third-party-prebuilt/*.whl; do \
            [ ! -f "$wheel" ] || python3 -m pip install --no-deps "$wheel"; \
        done; \
    fi; \
    mkdir -p /opt/upgrade/runtime-vendors; \
    vendors=step5_ops_transformer; \
    if [ "$WITH_UNION_OPS" = 1 ]; then vendors="$vendors union_test_transformer custom_nn_nn"; fi; \
    for v in $vendors; do \
        if [ -d "$ASCEND_OPP_PATH/vendors/$v" ]; then cp -a "$ASCEND_OPP_PATH/vendors/$v" /opt/upgrade/runtime-vendors/; fi; \
    done; \
    rm -rf /opt/upgrade/third-party-prebuilt
# Record actual identities; only source/wheels/runtime payload cross the stage.
RUN set -e; source /opt/upgrade/cann92-build-env.sh; \
    python3 -c 'import json,os,pathlib; c=pathlib.Path("/opt/upgrade"); d={"build_mode":"cann-base-a5-multistage","source_commit":os.environ["VLLM_ASCEND_COMMIT"],"version":os.environ["VLLM_ASCEND_VERSION"],"base_version":os.environ["VLLM_ASCEND_BASE_VERSION"],"base_image":os.environ["BASE_IMAGE"],"vllm_commit":os.environ["VLLM_COMMIT"],"vllm_version":os.environ["VLLM_VERSION"]}; (c/"evidence/SOURCE-IDENTITY.json").write_text(json.dumps(d,indent=2)+"\n")'; \
    python3 -m pip freeze > /opt/upgrade/evidence/pip-freeze.txt; \
    rm -rf /opt/upgrade/src/vllm/.git /opt/upgrade/src/vllm/build /opt/upgrade/src/vllm/dist /opt/upgrade/src/vllm/rust/target

# The final image inherits the CANN/system stage, never the compiler stage.
FROM a5-system AS runtime
ARG BASE_IMAGE
ARG VLLM_ASCEND_VERSION
ARG VLLM_ASCEND_COMMIT
ARG VLLM_VERSION=
ARG VLLM_COMMIT
ENV PATH=/opt/upgrade/.venv/bin:/opt/upgrade/cann-cli/bin:${PATH} \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=0 \
    VLLM_PLUGINS=ascend,ascend_model,ascend_kv_connector,ascend_model_loader,stepcast_platform,stepcast_rs
COPY --from=builder /opt/upgrade/.venv /opt/upgrade/.venv
COPY --from=builder /opt/upgrade/src/vllm /opt/upgrade/src/vllm
COPY --from=builder /opt/upgrade/src/vllm-commit.txt /opt/upgrade/src/vllm-version.txt /opt/upgrade/src/
COPY --from=builder /opt/upgrade/evidence /opt/upgrade/evidence
COPY --from=builder /opt/upgrade/runtime-vendors /opt/upgrade/runtime-vendors
COPY normalize-cann92.py cann92-env.sh cann92-build-env.sh cann92-entrypoint.sh probe_environment.py /opt/upgrade/
RUN set -e; python3 /opt/upgrade/normalize-cann92.py; source /opt/upgrade/cann92-env.sh; \
    mkdir -p "$ASCEND_OPP_PATH/vendors" /opt/upgrade/src/vllm-ascend; \
    for v in /opt/upgrade/runtime-vendors/*; do [ ! -d "$v" ] || cp -a "$v" "$ASCEND_OPP_PATH/vendors/"; done; \
    cp /opt/upgrade/evidence/SOURCE-IDENTITY.json /opt/upgrade/src/vllm-ascend/SOURCE-IDENTITY.json; \
    printf '#!/bin/bash\nsource /opt/upgrade/cann92-env.sh\n' > /opt/upgrade/runtime-env.sh; \
    chmod +x /opt/upgrade/runtime-env.sh; rm -rf /opt/upgrade/runtime-vendors; \
    python3 /opt/upgrade/probe_environment.py > /opt/upgrade/evidence/runtime.json
LABEL step5.build.mode="native" \
      step5.build.recipe="cann-base-a5-multistage" \
      step5.vllm-ascend.version="${VLLM_ASCEND_VERSION}" \
      step5.vllm-ascend.commit="${VLLM_ASCEND_COMMIT}" \
      org.stepfun.step5.source_base="${VLLM_ASCEND_COMMIT}" \
      step5.vllm.version="${VLLM_VERSION}" step5.vllm.commit="${VLLM_COMMIT}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" step5.cann.version="9.2.0"
WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "/opt/upgrade/cann92-entrypoint.sh"]
CMD ["/bin/bash"]
