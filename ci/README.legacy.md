# ci/ — Step5 / Ascend 950DT 镜像构建 CI 目录

本目录是 `vllm-ascend` 仓库（分支 `step5-swf-journal-ascend-ops`）内**独立**的 CI 目录，
把原先散落在交付目录 `/data/chensiyu/hw_project/step4/950_build/` 各处的
**构建脚本、产物索引、依赖索引、来源锁**统一收敛到版本库内，并提供一键构建入口
`./build.sh`。

设计原则：

1. **脚本与小配置文件入库，大二进制按来源分流**。构建真正必需的源（`source.bundle`）
   和小产物（cann-ops、cann-platform 元数据）vendor 进 `ci/inputs/`（约 88M）；
   其余大文件全部可从公开/内部 URL 获取（PyPI/OBS/GitHub/gitcode/quay.io，均带
   sha256 或 commit 锁定），获取方式登记在 `ci/inputs/REMOTE-SOURCES.md`——
   **不是每个人都有交付目录 `/data/chensiyu/...` 的访问权**。
2. **封存件逐字节一致**。`docker/Dockerfile`、`pins/*.json`、`scripts/candidate_probe.py`
   均从交付 `build/` 逐字节拷贝，并与 `pins/recipe-origin.json` 的 `context_files`
   SHA256 一一对应（见下表）；`runtime/*.sh` 是镜像内 `/opt/step5/bin/` 脚本的参考
   副本（由父镜像提供、Dockerfile 只 `source` 不 COPY），不参与 context pin。
3. **context 永远生成在仓库与交付之外**，遵循交付 `build/README.md` 的纪律：
   不在交付内构建、不生成缓存、不创建/重配置 daemon、不触网。
4. **两种构建后端**：`buildctl`（原有 BuildKit+containerd，aarch64 构建机，默认）与
   `docker`（buildx，任意 docker 主机；x86 经 QEMU 模拟 arm64，仅应急）。
5. **两条构建路径**：完整构建 `./build.sh`（native 重编译，权威配方）与快速构建
   `./build_fast.sh`（Python 级变更直接源码替换，分钟级；触及 csrc/二进制自动门禁拒绝）。

## 目录结构

```
ci/
├── build.sh                     # 一键构建入口（完整构建，native 重编译）
├── build_fast.sh                # 快速构建入口（= ./build.sh --fast，源码替换）
├── README.md                    # 本文件
├── .gitignore                   # 忽略 CI 运行产物（回执/日志/context/__pycache__）
├── config/
│   ├── delivery.env             # 交付资产路径（相对 CI_DELIVERY_ROOT，可覆盖）
│   └── build.params             # 构建默认参数（平台/target/后端/地址/namespace）
├── docker/
│   ├── Dockerfile               # 完整构建配方（= 交付 build/Dockerfile，逐字节）
│   └── Dockerfile.fast          # 快速构建配方（源码替换，不动封存配方）
├── pins/                        # 来源锁与封存锚点（均逐字节拷贝自交付）
│   ├── source-input.json        #   commit/tree/base/wheel/bundle sha pin
│   ├── feature-source-record.json
│   ├── recipe-origin.json       #   六件套 context SHA + wheel SHA + base manifest
│   └── source-lock.json         #   非 Ascend 源码锁
├── inputs/                      # vendor 进仓库的构建输入（约 88M）
│   ├── source.bundle            #   Git 恢复 bundle（pin 复现必需，79M）
│   ├── cann-ops/                #   算子 run/whl 产物 + ops-artifacts.json（8.4M）
│   ├── cann-platform/           #   CANN9.2 平台层元数据 + 2 个小 layer + 升级脚本
│   ├── source-assets-locks/     #   ascend/ops-thirdparty 的 URL+sha 归档锁
│   └── REMOTE-SOURCES.md        #   未入库大文件的完整 URL 获取清单（带 sha 锁定）
├── scripts/
│   ├── lib.sh                   # 公共函数（日志/校验/sha256/json 读取）
│   ├── check-assets.sh          # vendor 输入强制校验 + 交付资产（存在才）校验
│   ├── prepare-context.py       # 从本仓库生成六件套构建 context（pin/commit 两模式）
│   ├── import-oci.py            # OCI 导入现有 containerd namespace（先校验后导入）
│   ├── ensure-base-image.sh     # 确保后端存在 base 镜像（按 digest 精确匹配）
│   ├── run-build.sh             # buildctl / docker 两种后端执行构建
│   ├── verify-build.py          # 构建后校验 + 生成回执 JSON
│   ├── fast_replace.py          # 快速路径：镜像内源码替换 + 双重门禁
│   └── candidate_probe.py       # 容器内 CPU identity proof（拷贝自交付 inputs）
└── runtime/
    ├── step5-build-env.sh       # 构建期环境（HAL stub 等，拷贝自交付）
    └── step5-env.sh             # 运行期环境（拷贝自交付）
```

## 资产分流总表

| 资产 | 大小 | 去向 |
|---|---|---|
| `inputs/source.bundle` | 79M | **入库**（vendor，pin 复现必需） |
| `inputs/cann-ops/`（run+whl+json） | 8.4M | **入库**（vendor） |
| `inputs/cann-platform/` 元数据 + 2 个小 layer | ~10K | **入库**（记录 4.37G 层的 URL 复现方法） |
| `inputs/source-assets-locks/` | ~10K | **入库**（122M+11M 归档的 URL+sha 锁） |
| CANN 9.2 平台层 `9735e484….tar.gz` | 4.37G | **URL**：quay.io 官方镜像层（digest 锁定） |
| `wheelhouse/input/*.whl` ×8 | 623M | **URL**：PyPI/OBS/GitHub（`REMOTE-SOURCES.md` §3） |
| `source-assets/*`（4 个源码树） | 486M | **URL**：git 克隆 + 归档锁（§4） |
| `images/*.oci.tar`（parent+候选） | 16G | **容器仓库/交付目录**：base 镜像获取方式见 §5；推内部 registry 待定 |
| 封存 vllm_ascend wheel | 26M | 完整构建产物，任何人可重建 |

交付根目录可用环境变量 `CI_DELIVERY_ROOT` 或 `--delivery-root` 覆盖；交付目录**不存在时**
自动降级：仅用仓库内 vendor 输入 + URL 资产（check-assets 会打印跳过原因）。

## 一键构建

```bash
cd <repo>/ci
./build.sh                    # 复现封存 r2 镜像（默认 pin commit）
./build.sh --dry-run          # 只预检 + 生成 context，不构建
./build.sh --commit HEAD      # 构建当前 checkout 的 HEAD（commit 模式）
./build.sh --deep-check       # 资产校验含 16GB OCI 全量 sha256
./build.sh --backend docker   # 使用 docker buildx 后端
./build.sh --import-base      # base 镜像缺失时自动从 OCI 导入
./build.sh --help
```

流程：**预检 → 资产校验 → 生成 context → 确保 base 镜像 → 构建 → 校验+回执**。
成功后在 `ci/` 目录生成 `ci-build-receipt.json`（可用 `--receipt` 改路径；该文件被
`ci/.gitignore` 忽略），内容含 commit/tree/镜像 digest/context 六件套 SHA/是否 pin 复现。

### buildctl 后端（默认，aarch64 构建机）

要求构建机上已有 BuildKit daemon 与 containerd，且 base 镜像在
`--namespace`（默认 `step5-fp8-review-20261014`）内。950DT（gpu-a950dt-0002）
一次性准备（不写入 CI，由运维执行）：

```bash
# 950DT 上（原交付工具链仍保留时）
sudo -n /var/tmp/step5-fp8-review-20261014/tools/buildkit-0.23.2/bin/buildkitd \
  --config /var/tmp/step5-fp8-review-20261014/buildkit-step5.toml \
  --addr unix:///var/tmp/step5-fp8-review-20261014/buildkit.sock &
```

然后在本目录：

```bash
CI_BUILDKIT_ADDR=unix:///var/tmp/step5-fp8-review-20261014/buildkit.sock \
CI_CTR_BIN=/opt/basemind/containerd/bin/ctr CI_CTR_SUDO="sudo -n" ./build.sh
```

（950DT 上 `ctr` 不在 PATH 且 containerd socket 需要 root，故显式指定二进制并加
`CI_CTR_SUDO="sudo -n"`；其他主机保持默认即可。）

`buildctl` 后端与交付原始构建逐参数一致：`dockerfile.v0` 前端、`target=candidate`、
`platform=linux/arm64`、`image-resolve-mode=local`、显式 `FEATURE_COMMIT/FEATURE_TREE`，
Dockerfile 内的 `--network=none` 与 CANN native cache mount 原样保留。

### docker 后端（应急）

任意有 docker 的主机可用；base 镜像缺失时 `--import-base` 会从
`images/*.oci.tar`（16GB）`docker load`。x86 主机经 QEMU 模拟 arm64 编译 native
wheel 会很慢，仅建议在无 aarch64 构建机时应急使用。

## 快速构建 build_fast（源码替换，跳过 native 重编译）

```bash
./build_fast.sh                 # 对默认 commit 做快速源码替换
./build_fast.sh --commit HEAD   # 对当前 checkout 快速替换
./build_fast.sh --dry-run       # 只预检 + context + 门禁（不构建）
```

适用场景：**没有大版本升级、变更仅为 Python 级**（`vllm_ascend/**/*.py` 与 tests）。
做法：从 parent 镜像起容器，把变更文件覆盖到 `/vllm-workspace/vllm-ascend`
source checkout 与已安装的 site-packages `vllm_ascend/`（保留已编译 `.so`），
再跑同一组 CPU 契约测试——分钟级完成，不碰 csrc 编译。

**双重门禁**（构建前 `build.sh --fast` + 镜像内 `fast_replace.py`）：只要
`changed_paths` 触及 `csrc/`、`setup.py`、`pyproject.toml`、`CMakeLists.txt`、
`.gitmodules` 或 `vllm_ascend/` 下非文本文件，立即拒绝并指引使用完整构建
`./build.sh`。封存 commit 的 20 个变更文件全部是 Python，天然适用快速路径。

快速产物与完整构建分离：默认 tag `step5-fp8-a5:fast-<commit12>`、回执
`ci/ci-build-receipt-fast.json`、镜像 label `step5.ascend.installation=fast-source-replacement`，
**不承诺与完整构建逐 bit 一致**；设备验收同样需单独执行。

## 复现性与边界

- **pin 模式**（默认）：`prepare-context.py` 从本仓库重新生成 `source.tar`
  （`git archive`，属性覆盖逻辑与原始构建一致），bundle 取自仓库内
  `inputs/source.bundle`（交付目录仅作 fallback），最终六件套 SHA 必须全部等于
  `pins/recipe-origin.json` 的 `context_files`，否则报错退出——即"context 与当年
  实际构建输入逐字节一致"。
- **commit 模式**（`--commit <sha|HEAD>`）：为任意本仓库 commit 重新生成
  bundle（含 `HEAD` ref，与 probe 的 `bundle_ref=HEAD` 消费一致）与
  `source-input.json`/`feature-source-record.json`（字段白名单重建，文件级 SHA 由
  仓库实时计算；`feature_record_sha256` 在 record 落盘后重算；`cpu_tests` 取原 pin
  CPU 契约集与新 commit 的交集），记录实际 SHA，不做封存 pin 断言。
- 重新编译的 wheel/镜像**不承诺逐 bit 相同**；设备验收（八卡 950DT）需另行执行，
  参照交付 `validation/` 与 `reports/experiments/mtpgraph-20261016/`。
- 本目录不托管 658GB 模型权重、不启动任何 daemon、不触网；`--network=none`
  由 Dockerfile 保证构建期无外网。
- `scripts/check-assets.sh`、`ensure-base-image.sh`、`run-build.sh` 也可独立运行
  （自动推导 pins/仓库路径并按需加载 config；由 build.sh 调用时重复加载是幂等的）。
- `import-oci.py` 相较交付原版弱化了一处校验：只按 `final-manifest.json` 校验 OCI
  文件级 sha256/bytes，未交叉校验 `candidate-archive.json` 的 image manifest/config
  digest（原版逻辑仍在交付 `build/import-oci.py`，需要时可用它）。

## 多 agent review 记录

R1（2026-10-20，三路并行只读 review：bash 流程 / Python context / 文档与资产一致性）：

| 范围 | 结论 | 处理 |
|---|---|---|
| pin 模式六件套复现 | 实测 6/6 命中 recipe-origin.json，异常与安全设计健全 | 通过 |
| buildctl base 镜像匹配 | blocker：digest 前缀导致永远误判缺失 | 已修（digest 唯一匹配） |
| `--delivery-root` 覆盖顺序 | major：晚于 delivery.env source，资产校验指向错交付 | 已修（override 前移） |
| `ls\|wc -l` + pipefail | major：资产缺失时静默 exit 2 | 已修（nullglob 数组） |
| ctr sudo / 绝对路径 | major：非 root 与裸名 ctr 断裂 | 已修（CI_CTR_SUDO + command -v 解析） |
| commit 模式 record sha | blocker：`feature_record_sha256` 未重算，容器内 probe 必失败 | 已修（先落盘 record 再注入 sha） |
| `--base` 时 base_commit 不一致 | major：provenance 失真 | 已修（白名单重建同步 base_commit） |
| OCI 唯一匹配 | major：glob 取第一个、空匹配崩栈 | 已修（sorted + 恰好一个断言） |
| 其余 minor/nit（--help、取值校验、trap 现场保留、ctr stderr、attestation、README 口径等） | 18 项 | 已修 |

## 验证状态（2026-10-20）

| 环境 | 项目 | 结果 |
|---|---|---|
| 跳板机（x86_64） | pin 模式六件套 SHA 复现（`--dry-run`） | 6/6 命中 recipe-origin.json |
| 跳板机 | commit 模式（HEAD~1）bundle/record/pin 重建 | bundle 含 `HEAD` ref、record sha 自洽、字段白名单无泄漏 |
| 跳板机 | 资产校验 / 错误路径 / 独立运行 / `--delivery-root` 覆盖 / import-oci dry（16GB SHA） | 全部通过 |
| 950DT（aarch64） | `ensure-base-image.sh` 对真实 `ctr images ls` 的 digest 匹配、sudo+绝对路径 | 通过（base digest `c139edab…` 与 pin 一致） |
| 950DT | buildkit 0.23.2 工具链 | 在位（`/var/tmp/step5-fp8-review-20261014/tools/`） |
| 950DT | **完整 buildctl 构建** | **未验证**：buildkitd 当前未运行（需一次性启动，见上文）；`/data` 未挂载到 950DT，完整 `./build.sh` 需先把 context/交付资产带到该机；构建耗时以小时计，执行前请评估机器占用 |
