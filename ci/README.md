# TailRing Step5 / Ascend950DT 镜像 CI

## 当前默认实现（2026-10-23）

`ci/build.sh` 默认使用 `Dockerfile.tailring-source-rebase`，不再使用下文的历史源码：

- 分支：`step5-preview-tailring-npu`。
- vLLM：`ca5e43ddcd5b04c91e3de451702eecc1632faea8`。
- vLLM-Ascend：`143f3649d8181edae635091c5fe809eb51ff992c`，
  已合入 FULL FIA pre-builder padding 和 eager real-prefix 的 MTP 修复，并保留新 API 适配。
- Catlass：跳转机本地 CANN9.2 适配 `038b9315f3dd73b0a21200ee95fbaf314a1a9a1b`。

默认构建保留已验证的正式 CANN9.2/Flash/Step5 算子环境，
删除旧 vLLM 和 vLLM-Ascend 源码树，重新安装 vLLM 并全量重编译 Ascend native/ACLNN。
这不是跨 CANN 复用 native，也不代表重新编译了未改变的 Flash/Step5 外部算子。
新构建使用 `uv` 管理的 `/opt/upgrade/.venv`。

```bash
bash ci/build.sh --check-only --inputs-dir /data/chensiyu/step5-tailring-build-20261023/inputs
bash ci/build.sh --inputs-dir /data/chensiyu/step5-tailring-build-20261023/inputs \
  --context-dir /data/chensiyu/step5-tailring-context-20261023-new \
  --image quay.io/ascend/vllm-ascend:v0.30.0rc1-a5-openeuler-step5-cann9.2.0-tailring-143f3649d8 --keep
```

归档为已提交源码快照，不含 CI 缓存、`.git`、旧 `.so` 或未提交代码；
每份源码带 `SOURCE-IDENTITY.json`，且输入、配方和 helper 都有 checksum 锁。
新 Python 通用依赖按新源码显式 --no-deps 安装，保留已验证传递依赖，torch2.10/torch-npu/CANN 保持不变；
依赖声明适配有镜像内审计 ledger。`pip check` 结果以真实 receipt 为准。

完整官方基础配方仍保留在 `Dockerfile.official-v030-cann92`，源码锁已更新，
本次先全量编译624cc9d19a，再以Dockerfile.tailring-mtp-builder安装仅Python差异的143f3649d8；1114个native文件摘要未变。最终镜像config为98c123dc3ca238bf22970447b67747c6157a33c932de7c82ba9a63020ecfd4fe，安装wheel元数据仍标记624cc9d19a，实际Python源码143f3649d8单独记录在镜像ledger中。默认source-rebase配方已更新到143f，可完整重编译最新wheel。
镜像 build receipt 不等于 NPU 或四机 DP8 TP4 EP32 验收。

## 以下为历史 session 方案说明

默认入口现为 DSH `session-b5f168ee-ca7a-4022-a461-46e1867b75cb` 的最终完整配方，
不是旧的 v0.27 基础 candidate。CANN 归档现已改为版本可读文件名；
除文件名与来源注释外，session 的 payload 内容和构建步骤不变。`docker/Dockerfile`、`Dockerfile.fast` 与
`Dockerfile.official-v030-cann92` **彼此逐字节相同**；摘要见
`pins/official-v030-cann92.json`。小型 helper 在 `runtime/official-v030-cann92/`。

## 版本与构建纪律

- 官方 base：`quay.io/ascend/vllm-ascend:v0.30.0rc1-a5-openeuler`；构建前检查
  ARM64 config digest `sha256:c9d3e4921ba2153a0b1cec0005d01c76f7acce3def96cd6bca20a185a54f94d9`。
- CANN：**9.2.0 正式版**，payload SHA256 `6068329c27a345701d5cb7c1f76b7264f2c3320df9cfe2f8673ff37186f45c8f`。
- 保留 base 的 Python 3.12.13、torch 2.10.0+cpu、torch-npu 2.10.0.post4；
  Triton Ascend `3.6.0+dev20261013182112`，下载后校验 wheel 摘要。
- 使用 session 封存源码归档：vLLM `e177c621f2b8ae1be06190bc34eef84646e54f11`，
  vLLM-Ascend `317c7fb99e`；不是宣称源码本身等于官方 v0.30.0rc1。
- vLLM empty editable；vLLM-Ascend、Flash、Step5 算子在 CANN9.2 内全量重编译。
- 保留依赖声明适配、无驱动构建 HAL helper、Flash HCC 头文件检测与审计 ledger。
  **运行服务只 source `cann92-env.sh`，不能 source `cann92-build-env.sh`。**
- 上一次 ignore 漏掉最终新增的 `patch_flash_hcc.py`；本次 context 白名单已补齐。

## 准备输入与预检

默认归档目录为可版本化共享的 `ci/cache/official-v030rc1-cann9.2.0-317c7fb99e/`。也可用
`--inputs-dir` 或 `CI_UPGRADE_INPUTS_DIR` 指定。六个归档的大小和 SHA256 全部锁定，
脚本不会使用未提交源码、旧 native wheel、live HEAD 或静默跳过缺失输入。

```bash
bash ci/build.sh --check-only
# 报告 all_inputs_available=false 时只是记录现状，不代表构建可执行。
# 正式 CANN layer 未在跳转机上存在时，显式获取 4.8GB 官方锁定 layer：
# 共享缓存的大文件以 Git LFS 管理；发布者须先上传对象，其他人才能拉取。
git lfs pull internal --include='ci/cache/**'
# 尚未上传 CANN 对象时，可直接从带版本的官方镜像标签解析下载：
python3 ci/scripts/fetch-cann92-payload.py \
  --image quay.io/ascend/cann:9.2.0-950-openeuler24.03-py3.12-devel
# context 必须在仓库与交付目录之外；至少约 5.2GB 空间，跳转机 /tmp 可能不足。
mkdir -p /data/chensiyu/step5-ci-work
bash ci/build.sh --dry-run --context-dir /data/chensiyu/step5-ci-work/context-rc1-cann92
```

`--check-only` 校验所有存在的文件并报告缺失项，不下载、不构建、不接触 daemon。
`--dry-run` 必须具备全部输入；生成独立复制、二次 SHA 校验的 context 与回执。
实际源码固定到 session commit；`--commit HEAD` 只有 HEAD 仍为 `317c7fb99e` 时接受。
若将来更换源码，应另外更新归档锁、版本与验收，不能假称与本 session 相同。

## 实际构建

在 **native aarch64** 主机执行；x86 跳转机仅做检查/context 准备。本次新入口不会
为 QEMU/native 升级创建 daemon，不会动宿主驱动、权重、已有容器或服务。
先自行拉取/校验导入官方 base（本地标签必须有上述 config digest）：

```bash
bash ci/build.sh --backend docker \
  --inputs-dir /data/upgrade-v030-20261022 \
  --context-dir /data/chensiyu/step5-ci-work/context-new \
  --image step5-fp8-a5:ci-official-v030rc1-cann9.2.0-317c7fb99e --keep
```

入口使用 session 的 `DOCKER_BUILDKIT=0 docker build --network host` 与 DNS 参数，
Dockerfile 没有 `candidate`/`development` 命名 target，不能套用旧 target 参数。
默认 tag 与 session 验证镜像分开，避免覆盖。没有 registry push。
构建后镜像级检查会写 `ci-build-receipt-official-v030-cann92.json`；
**950DT 数值、推理和性能验收仍需另行执行**，不会用历史结果冒充新测试。

`--fast` / `build_fast.sh` 在新环境中安全降级为完整重编译，不复用旧 CANN native
产物。保留同名 Dockerfile 只是兼容调用路径，**并不是声称仍有 Python-only 快速路径**。

## 历史配方与回退

旧 Dockerfile 与全部旧 pins 未改身份，保存在 `Dockerfile.legacy`、
`Dockerfile.fast.legacy`、`build_legacy.sh` 与 `README.legacy.md`。
`config/build.params`、`config/delivery.env`、旧 `prepare-context.py`/
`candidate_probe.py`/`verify-build.py` 仅属于历史 candidate profile。

```bash
bash ci/build.sh --legacy --backend docker --dry-run
bash ci/build_fast.sh --legacy --backend docker --dry-run
```

## 证据与边界

session 成功镜像为本地 tag
`quay.io/ascend/vllm-ascend:v0.30.0rc1-a5-openeuler-step5-cann9.2.0-317c7fb99e`，
config `sha256:0a16c96cc8be9bacdd5028d35bed3623297b5313870d87229984c08062033483`。
它从真实 Flash 编译缓存 `sha256:1046b891679b3590163b7a25fbeabbd81fb29ce6b8ef7c862206034d80f152fd`
续建完成；最终修正后的完整配方没有从零重新执行，因此不保证重建 image ID 逐 bit 一致。
恢复配方仅作记录：`docker/Dockerfile.cann92-resume.reference`，不能作为默认基础。

session 的 `pip check` 有已记录冲突；保持同一依赖栈不是全面兼容性证明。
原 `probe_environment.py` 的 Git 字段指向 base 的 `/vllm-workspace`，不是实际
`/opt/upgrade/src` 定制源码；源码身份应以归档锁为准，不能误读该历史字段。
本次不改变这些 session helper，以保持配方/补丁行为一致。

Host-only 回归检查：

```bash
python3 -m unittest discover -s ci/tests -p 'test_official_session_build.py' -v
```

## 共享缓存

详见 `cache/README.md`。`cache/` 不整体忽略：清单和 README 作为普通 Git 文件，
大归档通过 `ci/.gitattributes` 使用 Git LFS；只忽略 `.part`、`.tmp` 和本地状态。
目录里的真实文件尚未 commit/push，也不代表 LFS 服务已经可用。
