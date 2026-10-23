# REMOTE-SOURCES — 未入库依赖的 URL 获取清单

本文件登记**没有**提交进本仓库的交付资产及其公开/内部 URL 来源。所有条目均带
sha256 或 git commit 锁定，可校验、可复现。仓库内只 vendor 了构建真正必需的
`ci/inputs/`（约 88M：source.bundle、cann-ops 产物、cann-platform 元数据），
其余大文件按本清单从 URL 获取——**不是每个人都有交付目录 `/data/chensiyu/...` 的访问权**。

约定：本文中的 `{sha}` 均已逐字节核对过交付目录中的原文件。

## 1. 本仓库已 vendor（ci/inputs/，无需 URL）

| 文件 | 大小 | sha256 |
|---|---|---|
| `inputs/source.bundle` | 82,666,269 B | `4b3cdb14bb9473cdc40d5606593034ba57db9cceb59eb4e14cb7bd36944b7291` |
| `inputs/cann-ops/cann-ops-transformer-step5_ops_linux-aarch64.run` | 8,424,828 B | `cda7c05a27b787491f1276d6a70ed7a43d738b7c9c6c0ac557a98b210095e922` |
| `inputs/cann-ops/cann_ops_transformer-1.0.0-py3-none-any.whl` | 581,763 B | `904c125e46707badaca21deedf9444b1ee7d2934c66fff73d09b685bea566ccd` |

## 2. CANN 9.2 平台层（4.37G，从 quay.io 官方镜像抽取）

官方镜像 `quay.io/ascend/cann:9.2.0-beta.2-950-openeuler24.03-py3.12-devel`
（manifest `sha256:0523f773498c81cf87b288b38c3c7626109463c798631d17d0ed9321c3c9fad0`，
完整 13 层清单见 `inputs/cann-platform/official-image-lock.json`）。

parent 镜像升级时只抽取其中 3 层（`selected_payload_layers`）：

| layer digest | 大小 | 位置 |
|---|---|---|
| `sha256:9735e4842a98fd25d906d3bd349b3461fe5c3e631d485101861295ee325a98da` | 4,691,731,401 B | **URL 获取**：`/usr/local/Ascend` 整层 |
| `sha256:5f0fafc82f4b5f2deb0c1aeb4f3da674af66f014e8479f5f9961cdfc88516ae2` | 222 B | 已入库 `inputs/cann-platform/` |
| `sha256:f7145efeeea1c8e065d9d8a298b3827ee6371c0059358170590154141f28cbe1` | 1,407 B | 已入库 `inputs/cann-platform/` |

复现方法（在 PR61/CANN9.1 基础镜像内执行，见 `inputs/cann-platform/upgrade-platform.sh`）：

```bash
skopeo copy docker-archive:///dev/stdin oci:/tmp/cann:9.2.0-beta.2  # 或 docker pull 后 save
# 按 official-image-lock.json 的 manifest 校验后，抽取 9735e484… 层：
curl -sL -H "Accept: application/vnd.oci.image.layer.v1.tar+gzip" \
  "https://quay.io/v2/ascend/cann/blobs/sha256:9735e4842a98fd25d906d3bd349b3461fe5c3e631d485101861295ee325a98da" \
  -o 9735e484….tar.gz
bash upgrade-platform.sh   # 内部先跑 verify-platform-layers.py 校验全部 digest
```

该层已完整烤进 parent 镜像 `step5-fp8-a5:modelslim-a2ac-wheel-20261016`；只有从零
重建 parent 时才需要单独获取。

## 3. wheelhouse 输入 wheel（623M，全部 URL 可拉取）

parent 镜像的固定输入。逐个数位锁定（`{sha}` 为交付原件 sha256）：

| wheel | 大小 | 来源 URL | sha256 |
|---|---|---|---|
| `build-1.6.1-py3-none-any.whl` | ~150K | PyPI：`https://pypi.org/simple/build/` | `ecd351a4be9d35a9eaaba244a7687143c9c7d4aea6ac964e7e7ddab20cbcf4e7` |
| `expecttest-0.3.0-py3-none-any.whl` | ~30K | PyPI：`https://pypi.org/simple/expecttest/` | `60f88103086e1754240b42175f622be83b6ffeac419434691ee5a5be819d0544` |
| `fastokens-0.2.0-cp39-abi3-manylinux_2_28_aarch64.whl` | ~2M | PyPI：`https://pypi.org/simple/fastokens/` | `b61b9fe5b41e0bb36ad86e7551dc53293c9833909ef07b1cdbaa2055b06c3b3e` |
| `flash_linear_attention_npu_a5-26.9.1+deva4a7958-py3-none-manylinux_2_34_aarch64.whl` | ~17M | GitHub 项目 `https://github.com/flashserve/flash-linear-attention-npu` 的 dev 构建产物；**精确版本 `+deva4a7958` 不在当前 releases**（现为 `+dev5cc6990` 及更新），需从对应 commit `a4a7958` 源码构建或向维护者索取 | `014ccd0cce4f21f4b030341a42e6be3065c2e6a436e7595e629de9d5c0547c48` |
| `pyelftools-0.32-py3-none-any.whl` | ~180K | PyPI：`https://pypi.org/simple/pyelftools/` | `013df952a006db5e138b1edf6d8a68ecc50630adbd0d83a2d41e7f846163d738` |
| `pyproject_hooks-1.3.0-py3-none-any.whl` | ~12K | PyPI：`https://pypi.org/simple/pyproject-hooks/` | `733909c6ac133d222c0b8cfc11364f19d363793d8fbcd8bb7240c5220af8d35d` |
| `triton-3.6.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl` | ~110M | PyPI：`https://pypi.org/simple/triton/`（精确文件名已在索引中验证） | `374f52c11a711fd062b4bfbb201fd9ac0a5febd28a96fb41b4a0f51dde3157f4` |
| `triton_ascend-3.6.0+dev20261013182112-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl` | ~200M | 华为 OBS（已验证 HTTP 200）：`https://ascend-triton-open.obs.cn-north-4.myhuaweicloud.com/triton-ascend/20261013182112/triton_ascend-3.6.0%2Bdev20261013182112-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl` | `d46b9b80a3e0cb21e1965410acacf800b47bd16a150e995f06194b01aaf59c03` |

## 4. 算子/三方源码（486M，git URL + 归档锁）

| 目录 | 来源 | 锁定 |
|---|---|---|
| `flash-attention-npu` | `https://github.com/MinghuasLab/flash-attention-npu.git` | commit `7f12d09eb7bed5c8412617019ba94e63e3a0fe2d`（branch `step5-cann92-static-vf`，即 tag `v0.4.0-1-g7f12d09`） |
| `ops-transformer` | `https://gitcode.com/HuangKun8682/ops-transformer_inedex_topk.git` | commit `b24ae2534fa11e95bcacca37c266659c579a465b`（branch `stepx`） |
| `ascend-thirdparty`（11M） | gitcode releases，逐文件 URL+sha 见 `inputs/source-assets-locks/ascend-thirdparty-archive-lock.json` | 清单内含全部 sha256 |
| `ops-thirdparty`（122M） | gitcode + 华为 OBS `cann-3rd`，逐文件 URL+sha 见 `inputs/source-assets-locks/ops-thirdparty-archive-lock.json` | 清单内含全部 sha256 |

## 5. 基础镜像（parent 的 FROM，URL/内部镜像）

| 用途 | 引用 |
|---|---|
| parent 镜像的 base（PR61 配方 `BASE_IMAGE`） | `quay.io/ascend/vllm-ascend:v0.27.1rc1-a5-openeuler`；内部镜像：`artifactory.stepfun-inc.com/docker-quayio/ascend/vllm-ascend@sha256:6b98274b9976afb1a4fa15caca0cd2d007aae111ebd278e8bf1736674d96eec3` |
| 本 CI 的 base（候选镜像的 FROM） | `step5-fp8-a5:modelslim-a2ac-wheel-20261016@sha256:c139edabf0467dfaa0063eabf7e2d4cd229439b362525e8da5b21bd1871f6f4c`——**构建产物**，获取方式：① 950DT containerd namespace `step5-fp8-review-20261014`（已在位）；② 交付目录 `images/*.oci.tar`（16G，含 parent+候选，`ctr images import` 或 `docker load`）；③ 待推内部 GitLab 容器仓库（尚未推送，见 ci/README.md） |

## 6. 主源码（git）

| 仓库 | 分支/commit | 位置 |
|---|---|---|
| vllm-ascend（本仓库） | `step5-swf-journal-ascend-ops` @ `7ae69ef36e5027a379ebbd1464503135a9a62034` | 本仓库；内部 GitLab `gitlab.basemind.com/sys/stepcast/multi-hardware/vllm-ascend.git` |
| vllm | `step-5-preview-tracking` @ `e177c621f2b8ae1be06190bc34eef84646e54f11` | 内部 GitLab `sys/stepcast/multi-hardware` 组（候选构建不直接需要：parent 镜像已含 editable 安装） |

## 7. 明确不入库的大文件

| 文件 | 大小 | 理由 |
|---|---|---|
| `cann-platform/payload/9735e484….tar.gz` | 4.37G | quay.io 官方镜像层，digest 锁定（§2） |
| `wheelhouse/input/*.whl` ×8 | 623M | PyPI/OBS/GitHub URL（§3） |
| `source-assets/*` | 486M | git/归档锁 URL（§4） |
| `images/*.oci.tar` | 16G | parent 为构建产物；base 镜像获取方式见 §5 |
