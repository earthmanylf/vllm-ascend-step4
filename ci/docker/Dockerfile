ARG BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.30.0rc1-a5-openeuler
FROM ${BASE_IMAGE}
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV TORCH_DEVICE_BACKEND_AUTOLOAD=0 SOC_VERSION=ascend950dt_9582 MAX_JOBS=16 CARGO_BUILD_JOBS=16
# vllm-ascend is compiled from the archive the build context carries, which is
# exported from the current HEAD. The image has no .git, so setup.py cannot
# derive a version: the build passes it in. Default keeps an unversioned build
# obviously unversioned instead of silently claiming a pinned one.
ARG VLLM_ASCEND_VERSION=0.0.0
ARG VLLM_ASCEND_COMMIT=unknown
# Some base images already carry the CANN 9.2.0 toolchain, the vendored
# third-party sources and the compiled custom ops. Rebuilding them needs a 4.8 GB
# payload plus three archives; reusing them needs none of that and makes the
# native build possible offline. Set to 1 when the base already has them.
ARG REUSE_BASE_ASSETS=0
# Union-mode sparse attention needs two ops that the released packages do not
# ship (they reject maskType=3 outright): the union GBSA kernel and
# UnionSparseMeta. Both are built outside this recipe because they need a
# toolchain run per op repo, and the result is installed from flattened vendor
# trees staged at third-party/prebuilt/. Set to 1 to install them.
ARG WITH_UNION_OPS=0
# CANN 9.2.0 formal release from quay.io/ascend/cann:9.2.0-950-openeuler24.03-py3.12-devel
COPY cann92-release/cann-9.2.0-950-openeuler24.03-py3.12-devel-ascend.tar.gz /opt/upgrade/cann92-release/cann-9.2.0-950-openeuler24.03-py3.12-devel-ascend.tar.gz
RUN set -euo pipefail; \
    if [ "${REUSE_BASE_ASSETS}" = "1" ]; then \
        test -d /usr/local/Ascend/cann-9.2.0 || { echo "REUSE_BASE_ASSETS=1 but the base has no CANN 9.2.0" >&2; exit 1; }; \
        echo "reusing CANN 9.2.0 from the base image"; \
        rm -f /opt/upgrade/cann92-release/cann-*.tar.gz; \
    else \
        echo '6068329c27a345701d5cb7c1f76b7264f2c3320df9cfe2f8673ff37186f45c8f  /opt/upgrade/cann92-release/cann-9.2.0-950-openeuler24.03-py3.12-devel-ascend.tar.gz' | sha256sum -c -; \
        tar -xzf /opt/upgrade/cann92-release/cann-9.2.0-950-openeuler24.03-py3.12-devel-ascend.tar.gz -C /; \
        test -d /usr/local/Ascend/cann-9.2.0; \
    fi
COPY cann92-env.sh /opt/upgrade/cann92-env.sh
COPY normalize-cann92.py /opt/upgrade/normalize-cann92.py
RUN python3 /opt/upgrade/normalize-cann92.py
# vllm comes from git, not from a pinned tarball: a build then always reflects a
# ref, and the image tag can name the commit it actually contains.
#
# The credential is a build-arg with no default on purpose -- it must not live in
# the repository. The clone URL is assembled at build time and .git is deleted
# afterwards so the token is not baked into the image layers.
ARG VLLM_SOURCE=git
ARG VLLM_REPO=https://gitlab.basemind.com/sys/stepcast/vllm.git
ARG VLLM_TAG=step5-preview-tailring-npu
ARG VLLM_COMMIT=
ARG GITLAB_ACCESS_TOKEN=
# The vllm series this image is built for. The fork's own tags stop at
# v0.26.1rc0, so deriving from `git describe` alone would label a 0.30 image as
# 0.26.1; anchor on this instead (same policy as VLLM_ASCEND_VERSION).
ARG VLLM_BASE_VERSION=0.30.0rc1
ARG VLLM_VERSION=
RUN set -euo pipefail; \
    command -v git >/dev/null 2>&1 || { echo "git is required in the base image" >&2; exit 1; }; \
    export GIT_TERMINAL_PROMPT=0; \
    if [ "${GITLAB_ACCESS_TOKEN}" != "" ]; then \
        url="$(printf '%s' "${VLLM_REPO}" | sed "s|https://|https://oauth2:${GITLAB_ACCESS_TOKEN}@|")"; \
    else \
        url="${VLLM_REPO}"; \
    fi; \
    echo "vllm source: ${VLLM_SOURCE} commit=${VLLM_COMMIT:-<none>} tag=${VLLM_TAG}"; \
    previous="/opt/upgrade/src/vllm/SOURCE-IDENTITY.json"; \
    old_commit=""; old_version=""; \
    if [ -f "$previous" ]; then \
        old_commit="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("commit",""))' "$previous" 2>/dev/null || true)"; \
        old_version="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("source_version",""))' "$previous" 2>/dev/null || true)"; \
    fi; \
    rm -rf /opt/upgrade/src/vllm; \
    mkdir -p /opt/upgrade/src; \
    if [ "${VLLM_COMMIT}" != "" ]; then \
        git clone "${url}" /opt/upgrade/src/vllm; \
        git -C /opt/upgrade/src/vllm checkout --detach "${VLLM_COMMIT}"; \
    else \
        git clone --depth 1 "${url}" --branch "${VLLM_TAG}" /opt/upgrade/src/vllm; \
    fi; \
    commit="$(git -C /opt/upgrade/src/vllm rev-parse HEAD)"; \
    printf '%s\n' "$commit" > /opt/upgrade/src/vllm-commit.txt; \
    version="${VLLM_VERSION:-}"; \
    if [ -n "$old_commit" ] && [ "$old_commit" = "$commit" ] && [ -n "$old_version" ] && [ -z "$version" ]; then \
        version="$old_version"; \
        echo "vllm version: base image already recorded $version for this commit"; \
    fi; \
    if [ -z "$version" ]; then \
        version="0.0.0+g$(printf '%s' "$commit" | cut -c1-12)"; \
        echo "vllm version: VLLM_VERSION was not supplied, recording $version rather than inventing a series" >&2; \
    fi; \
    case "$version" in \
        *"$commit"*|*"$(printf '%s' "$commit" | cut -c1-12)"*) ;; \
        *) echo "vllm version $version does not name the cloned commit $commit" >&2; exit 1 ;; \
    esac; \
    echo "vllm version: $version for $commit (series ${VLLM_BASE_VERSION})"; \
    printf '%s\n' "$version" > /opt/upgrade/src/vllm-version.txt; \
    rm -rf /opt/upgrade/src/vllm/.git; \
    cat /opt/upgrade/src/vllm-commit.txt /opt/upgrade/src/vllm-version.txt
# Extract into an empty directory. ADD merges into whatever the base image left
# behind, so a module deleted upstream (e.g. models/step5/_decode_summary_layout.py)
# would survive in the source tree and be packaged into the wheel again.
COPY vllm-ascend.tar.gz /opt/upgrade/vllm-ascend-source.tar
# csrc/third_party is not part of the source export: catlass, abseil-cpp and
# ascend_protobuf come from the base image (the archive that would carry them is
# an empty placeholder under REUSE_BASE_ASSETS=1), so it has to survive the wipe.
RUN set -euo pipefail; \
    keep=/opt/upgrade/.vllm-ascend-third-party; \
    rm -rf "$keep"; \
    if [ -d /opt/upgrade/src/vllm-ascend/csrc/third_party ]; then \
        cp -a /opt/upgrade/src/vllm-ascend/csrc/third_party "$keep"; \
    fi; \
    rm -rf /opt/upgrade/src/vllm-ascend; \
    mkdir -p /opt/upgrade/src/vllm-ascend; \
    tar -xf /opt/upgrade/vllm-ascend-source.tar -C /opt/upgrade/src/vllm-ascend; \
    rm -f /opt/upgrade/vllm-ascend-source.tar; \
    if [ -d "$keep" ]; then \
        rm -rf /opt/upgrade/src/vllm-ascend/csrc/third_party; \
        cp -a "$keep" /opt/upgrade/src/vllm-ascend/csrc/third_party; \
        rm -rf "$keep"; \
    fi; \
    test -d /opt/upgrade/src/vllm-ascend/csrc/third_party/catlass/include || { echo "catlass/include missing after the source reset" >&2; exit 1; }; \
    echo "vllm-ascend source reset:" && ls /opt/upgrade/src/vllm-ascend/csrc/third_party
ADD ascend-catlass.tar.gz /opt/upgrade/src/vllm-ascend/csrc/third_party/catlass/
ADD custom-assets.tar.gz /opt/upgrade/
ADD ops-git-metadata.tar.gz /opt/upgrade/
# The base image already has these when REUSE_BASE_ASSETS=1; only repair what is
# missing, so the same recipe works for a full rebuild and an offline one.
RUN set -euo pipefail; \
    for d in /opt/upgrade/source-assets/ascend-thirdparty /opt/step5/ascend-thirdparty; do \
        [ -d "$d" ] && chown -R root:root "$d" || true; \
    done; \
    if [ "${REUSE_BASE_ASSETS}" = "1" ]; then \
        echo "reusing vendored third-party sources from the base image"; \
        rm -f /opt/upgrade/custom-assets.tar.gz /opt/upgrade/ops-git-metadata.tar.gz; \
        for c in json makeself pkg; do \
            tgt="/opt/upgrade/src/vllm-ascend/csrc/third_party/$c"; \
            if [ ! -e "$tgt" ]; then \
                src="/opt/step5/ascend-thirdparty/$c"; \
                [ -e "$src" ] || { echo "missing third-party $c" >&2; exit 1; }; \
                ln -s "$src" "$tgt"; \
            fi; \
        done; \
        test -d /opt/upgrade/src/vllm-ascend/csrc/third_party/catlass || { echo "missing catlass" >&2; exit 1; }; \
    else \
        ln -sfn /opt/upgrade/source-assets/ascend-thirdparty/json /opt/upgrade/src/vllm-ascend/csrc/third_party/json; \
        ln -sfn /opt/upgrade/source-assets/ascend-thirdparty/makeself /opt/upgrade/src/vllm-ascend/csrc/third_party/makeself; \
        ln -sfn /opt/upgrade/source-assets/ascend-thirdparty/pkg /opt/upgrade/src/vllm-ascend/csrc/third_party/pkg; \
    fi
COPY apply_dependency_profile.py probe_environment.py install_ops.py /opt/upgrade/
RUN mkdir -p /opt/upgrade/evidence && python3 /opt/upgrade/apply_dependency_profile.py && \
    python3 -m pip install build==1.6.1 expecttest==0.3.0 pyelftools==0.32 && \
    python3 -m pip install -r /opt/upgrade/src/vllm/requirements/common.txt -c /opt/upgrade/constraints-tailring.txt transformers==5.19.0 && \
    if python3 -c 'import importlib.metadata as m,sys; sys.exit(0 if m.version("triton-ascend") == "3.6.0+dev20261013182112" else 1)' 2>/dev/null; then \
        echo "triton-ascend 3.6.0+dev20261013182112 already in the base image; keeping it"; \
    else \
        python3 -m pip download --no-deps 'https://ascend-triton-open.obs.cn-north-4.myhuaweicloud.com/triton-ascend/20261013182112/triton_ascend-3.6.0%2Bdev20261013182112-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl' -d /opt/upgrade/wheels && \
        echo 'd46b9b80a3e0cb21e1965410acacf800b47bd16a150e995f06194b01aaf59c03  /opt/upgrade/wheels/triton_ascend-3.6.0+dev20261013182112-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl' | sha256sum -c - && \
        python3 -m pip uninstall -y triton triton-ascend && \
        python3 -m pip install --no-deps /opt/upgrade/wheels/triton_ascend-*.whl; \
    fi
RUN source /opt/upgrade/cann92-env.sh && python3 -m pip uninstall -y vllm && \
    CARGO_NET_OFFLINE=true SETUPTOOLS_SCM_PRETEND_VERSION="$(cat /opt/upgrade/src/vllm-version.txt)" VLLM_TARGET_DEVICE=empty \
    python3 -m pip install -e /opt/upgrade/src/vllm --no-build-isolation --no-deps
COPY cann92-build-env.sh /opt/upgrade/cann92-build-env.sh
RUN set -euo pipefail; \
    set +u; source /opt/upgrade/cann92-build-env.sh; set -u; \
    cd /opt/upgrade/src/vllm-ascend; \
    rm -f /opt/upgrade/wheels/vllm_ascend-*.whl; \
    COMPILE_CUSTOM_KERNELS=1 SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_ASCEND_VERSION}" \
        python3 -m pip wheel . --no-build-isolation --no-deps -w /opt/upgrade/wheels; \
    cd /tmp; \
    python3 -m pip uninstall -y vllm-ascend || true; \
    site="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    rm -rf "$site/vllm_ascend" "$site"/vllm_ascend-*.dist-info; \
    python3 -m pip install --no-deps /opt/upgrade/wheels/vllm_ascend-*.whl; \
    python3 -c 'import json,os,pathlib,sys; \
site=pathlib.Path(__import__("sysconfig").get_paths()["purelib"])/"vllm_ascend"; \
src=pathlib.Path("/opt/upgrade/src/vllm-ascend/vllm_ascend"); \
allow={"__init__.py"}; \
extra=sorted(str(p.relative_to(site)) for p in site.rglob("*.py") \
             if p.parent.name!="__pycache__" \
             and p.name not in ("_version.py","_build_info.py","version.py") \
             and not (src/p.relative_to(site)).is_file()); \
print("vllm_ascend stale modules shipped by the wheel:", extra) if extra else print("vllm_ascend: installed tree matches the source export"); \
sys.exit(1) if extra else None'; \
    python3 -c 'import json,os,pathlib; \
c=pathlib.Path("/opt/upgrade/src"); \
vllm_runtime=json.loads((c/"vllm/SOURCE-IDENTITY.json").read_text()) if (c/"vllm/SOURCE-IDENTITY.json").is_file() else {}; \
identity={"build_mode":"native","recipe":"ci/docker/Dockerfile","source_commit":os.environ["VLLM_ASCEND_COMMIT"],"version":os.environ["VLLM_ASCEND_VERSION"],"base_version":os.environ.get("VLLM_ASCEND_BASE_VERSION",""),"base_image":os.environ.get("VLLM_ASCEND_BASE_IMAGE",""),"vllm_commit":(c/"vllm-commit.txt").read_text().strip(),"vllm_version":(c/"vllm-version.txt").read_text().strip(),"vllm_runtime_commit":vllm_runtime.get("commit",""),"vllm_runtime_version":vllm_runtime.get("source_version","")}; \
(c/"vllm-ascend/SOURCE-IDENTITY.json").write_text(json.dumps(identity,indent=2)+"\n"); \
print(json.dumps(identity,indent=2))'
COPY patch_flash_hcc.py /opt/upgrade/patch_flash_hcc.py
# flash-attn-npu and the step5 ops are compiled from source. With
# REUSE_BASE_ASSETS=1 both are already installed in the base image, so this is
# skipped and verified instead: the whole point of that switch is to build
# offline, and these two steps are the parts that need the sources.
RUN set -euo pipefail; \
    if [ "${REUSE_BASE_ASSETS}" = "1" ]; then \
        echo "reusing flash-attn-npu and step5 ops from the base image"; \
        set +u; source /opt/upgrade/cann92-env.sh; set -u; \
        python3 -c 'import glob,importlib.metadata as m,sys; \
d=m.distribution("flash-attn-npu").locate_file(""); \
sys.exit(0 if glob.glob(str(d) + "/flash_attn_npu_*.so") else 1)' \
            || { echo "base lacks the flash_attn_npu extension" >&2; exit 1; }; \
        test -d "$ASCEND_OPP_PATH/vendors/step5_ops_transformer" || { echo "base lacks step5_ops_transformer" >&2; exit 1; }; \
        python3 /opt/upgrade/probe_environment.py > /opt/upgrade/evidence/development.json; \
    else \
        python3 /opt/upgrade/patch_flash_hcc.py; \
        source /opt/upgrade/cann92-build-env.sh; \
        cd /opt/upgrade/source-assets/flash-attention-npu; \
        FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_BUILD_NPU=950 python3 setup.py bdist_wheel; \
        cd /tmp; python3 -m pip install --force-reinstall --no-deps /opt/upgrade/source-assets/flash-attention-npu/dist/flash_attn_npu-*.whl; \
        cd /opt/upgrade/source-assets/ops-transformer; \
        bash build.sh --pkg --soc=ascend950 --vendor_name=step5_ops --ops=generic_block_sparse_attention,generic_block_sparse_attention_metadata,indexer_region_score --cann_3rd_lib_path=/opt/upgrade/source-assets/ops-thirdparty -j16; \
        bash build.sh --torch_extension && python3 /opt/upgrade/install_ops.py build_out; \
        cd /tmp; python3 /opt/upgrade/probe_environment.py > /opt/upgrade/evidence/development.json; \
    fi
# Union-mode custom ops. The vendor trees come pre-flattened: the .run
# installers add an extra vendors/<name> level, and python cannot see an op
# through that, so ci/third-party/fetch.py (or the operator) must flatten them
# before they are staged. Installing by copy keeps this step fast and avoids
# needing a compiler inside the image.
ARG WITH_UNION_OPS=0
COPY third-party/prebuilt/ /opt/upgrade/third-party-prebuilt/
RUN set -euo pipefail; \
    if [ "${WITH_UNION_OPS}" != "1" ]; then \
        echo "union ops: not requested"; \
        rm -rf /opt/upgrade/third-party-prebuilt; \
        exit 0; \
    fi; \
    set +u; source /opt/upgrade/cann92-env.sh; set -u; \
    OPP="$ASCEND_OPP_PATH"; \
    for v in union_test_transformer custom_nn_nn; do \
        src="/opt/upgrade/third-party-prebuilt/$v"; \
        [ -d "$src/op_api/lib" ] || { echo "union ops: $v is missing or not flattened (need $v/op_api/lib)" >&2; exit 1; }; \
        rm -rf "$OPP/vendors/$v"; \
        cp -a "$src" "$OPP/vendors/$v"; \
        echo "union ops: installed $v"; \
    done; \
    for whl in /opt/upgrade/third-party-prebuilt/*.whl; do \
        [ -e "$whl" ] || continue; \
        python3 -m pip install --no-deps "$whl" && echo "union ops: installed $(basename "$whl")"; \
    done; \
    python3 -c "import glob,subprocess,sys; \
libs=glob.glob('$OPP/vendors/*/op_api/lib/libcust_opapi.so'); \
missing=[l for l in libs if b'UnionSparseMeta' not in subprocess.run(['nm','-D','--defined-only',l],capture_output=True).stdout]; \
print('union ops: checked', len(libs), 'vendor libs'); \
sys.exit(1) if not any(b'UnionSparseMeta' in subprocess.run(['nm','-D','--defined-only',l],capture_output=True).stdout for l in libs) else None"; \
    rm -rf /opt/upgrade/third-party-prebuilt

COPY cann92-entrypoint.sh /opt/upgrade/cann92-entrypoint.sh
LABEL step5.vllm-ascend.version="${VLLM_ASCEND_VERSION}" \
      step5.vllm-ascend.commit="${VLLM_ASCEND_COMMIT}" \
      org.opencontainers.image.base.name="quay.io/ascend/vllm-ascend:v0.30.0rc1-a5-openeuler" \
      step5.cann.version="9.2.0" step5.cann.arm64.manifest="sha256:811ca9bb37475ba1a8db6cd1f52731651c13f5860a4cfb58773a32e603e02f57"
WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "/opt/upgrade/cann92-entrypoint.sh"]
CMD ["/bin/bash"]
