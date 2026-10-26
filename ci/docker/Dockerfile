ARG BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.30.0rc1-a5-openeuler
FROM ${BASE_IMAGE}
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV TORCH_DEVICE_BACKEND_AUTOLOAD=0 SOC_VERSION=ascend950dt_9582 MAX_JOBS=16 CARGO_BUILD_JOBS=16
# CANN 9.2.0 formal release from quay.io/ascend/cann:9.2.0-950-openeuler24.03-py3.12-devel
COPY cann92-release/cann-9.2.0-950-openeuler24.03-py3.12-devel-ascend.tar.gz /opt/upgrade/cann92-release/cann-9.2.0-950-openeuler24.03-py3.12-devel-ascend.tar.gz
RUN echo '6068329c27a345701d5cb7c1f76b7264f2c3320df9cfe2f8673ff37186f45c8f  /opt/upgrade/cann92-release/cann-9.2.0-950-openeuler24.03-py3.12-devel-ascend.tar.gz' | sha256sum -c - && \
    tar -xzf /opt/upgrade/cann92-release/cann-9.2.0-950-openeuler24.03-py3.12-devel-ascend.tar.gz -C / && \
    test -d /usr/local/Ascend/cann-9.2.0
COPY cann92-env.sh /opt/upgrade/cann92-env.sh
COPY normalize-cann92.py /opt/upgrade/normalize-cann92.py
RUN python3 /opt/upgrade/normalize-cann92.py
ADD vllm.tar.gz /opt/upgrade/src/vllm/
ADD vllm-ascend.tar.gz /opt/upgrade/src/vllm-ascend/
ADD ascend-catlass.tar.gz /opt/upgrade/src/vllm-ascend/csrc/third_party/catlass/
ADD custom-assets.tar.gz /opt/upgrade/
ADD ops-git-metadata.tar.gz /opt/upgrade/
RUN chown -R root:root /opt/upgrade/source-assets && \
    ln -s /opt/upgrade/source-assets/ascend-thirdparty/json /opt/upgrade/src/vllm-ascend/csrc/third_party/json && \
    ln -s /opt/upgrade/source-assets/ascend-thirdparty/makeself /opt/upgrade/src/vllm-ascend/csrc/third_party/makeself && \
    ln -s /opt/upgrade/source-assets/ascend-thirdparty/pkg /opt/upgrade/src/vllm-ascend/csrc/third_party/pkg
COPY apply_dependency_profile.py probe_environment.py install_ops.py /opt/upgrade/
RUN mkdir -p /opt/upgrade/evidence && python3 /opt/upgrade/apply_dependency_profile.py && \
    python3 -m pip install build==1.6.1 expecttest==0.3.0 pyelftools==0.32 && \
    python3 -m pip install -r /opt/upgrade/src/vllm/requirements/common.txt -c /opt/upgrade/constraints-tailring.txt transformers==5.19.0 && \
    python3 -m pip download --no-deps 'https://ascend-triton-open.obs.cn-north-4.myhuaweicloud.com/triton-ascend/20261013182112/triton_ascend-3.6.0%2Bdev20261013182112-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl' -d /opt/upgrade/wheels && \
    echo 'd46b9b80a3e0cb21e1965410acacf800b47bd16a150e995f06194b01aaf59c03  /opt/upgrade/wheels/triton_ascend-3.6.0+dev20261013182112-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl' | sha256sum -c - && \
    python3 -m pip uninstall -y triton triton-ascend && \
    python3 -m pip install --no-deps /opt/upgrade/wheels/triton_ascend-*.whl
RUN source /opt/upgrade/cann92-env.sh && python3 -m pip uninstall -y vllm && \
    CARGO_NET_OFFLINE=true SETUPTOOLS_SCM_PRETEND_VERSION=0.29.1rc1.dev2194+gca5e43ddcd VLLM_TARGET_DEVICE=empty \
    python3 -m pip install -e /opt/upgrade/src/vllm --no-build-isolation --no-deps
COPY cann92-build-env.sh /opt/upgrade/cann92-build-env.sh
RUN source /opt/upgrade/cann92-build-env.sh && cd /opt/upgrade/src/vllm-ascend && \
    COMPILE_CUSTOM_KERNELS=1 SETUPTOOLS_SCM_PRETEND_VERSION=0.19.1rc2.dev2977+g143f3649d8 \
    python3 -m pip wheel . --no-build-isolation --no-deps -w /opt/upgrade/wheels && \
    cd /tmp && python3 -m pip uninstall -y vllm-ascend && \
    python3 -m pip install --force-reinstall --no-deps /opt/upgrade/wheels/vllm_ascend-*.whl
COPY patch_flash_hcc.py /opt/upgrade/patch_flash_hcc.py
RUN python3 /opt/upgrade/patch_flash_hcc.py
RUN source /opt/upgrade/cann92-build-env.sh && cd /opt/upgrade/source-assets/flash-attention-npu && \
    FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_BUILD_NPU=950 python3 setup.py bdist_wheel && \
    cd /tmp && python3 -m pip install --force-reinstall --no-deps /opt/upgrade/source-assets/flash-attention-npu/dist/flash_attn_npu-*.whl
RUN source /opt/upgrade/cann92-build-env.sh && cd /opt/upgrade/source-assets/ops-transformer && \
    bash build.sh --pkg --soc=ascend950 --vendor_name=step5_ops --ops=generic_block_sparse_attention,generic_block_sparse_attention_metadata,indexer_region_score --cann_3rd_lib_path=/opt/upgrade/source-assets/ops-thirdparty -j16 && \
    bash build.sh --torch_extension && python3 /opt/upgrade/install_ops.py build_out && \
    cd /tmp && python3 /opt/upgrade/probe_environment.py > /opt/upgrade/evidence/development.json
COPY cann92-entrypoint.sh /opt/upgrade/cann92-entrypoint.sh
LABEL org.opencontainers.image.base.name="quay.io/ascend/vllm-ascend:v0.30.0rc1-a5-openeuler" \
      step5.cann.version="9.2.0" step5.cann.arm64.manifest="sha256:811ca9bb37475ba1a8db6cd1f52731651c13f5860a4cfb58773a32e603e02f57"
WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "/opt/upgrade/cann92-entrypoint.sh"]
CMD ["/bin/bash"]
