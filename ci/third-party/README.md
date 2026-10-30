# ci/third-party — union 算子的源码、依赖与产物

这里**只放清单和工具，不放源码**：两个算子仓各 300 MB 左右，它们的编译期依赖
还有 250 MB，CANN payload 4.8 GB，都不适合进 git。`fetch.py` 负责把源码落到本
目录并**逐个 ref 校验**，`build-union-ops.sh` 负责离线编译出两个算子包并压平到
`prebuilt/`。

```bash
cd ci/third-party
python3 fetch.py --dry-run           # 需要什么、当前缺什么
python3 fetch.py                     # 有网：git clone
python3 fetch.py --submodules        # 连 ops-tensor 的 tensor_api 子模块一起校验
./build-union-ops.sh                 # 编译 + 压平成 prebuilt/
```

退出码 0 表示全部就绪；非 0 会列出 `Blocked:` 项。

## 目录

```
ci/third-party/
├── README.md              本文件
├── manifest.json          每个源：repo / 分支 / 需要的 commit 或 tag / 用途
├── fetch.py               跳过 / 拷贝 / clone，三选一，都校验 ref
├── build-union-ops.sh     在容器里编译两个算子包，压平进 prebuilt/
├── ops-transformer/       union 版 GBSA 算子（maskType=3）
├── ops-nn/                UnionSparseMeta + torch extension
├── cann-cmake/  opbase/  ops-tensor/  catlass/  ops-test-kit/
│                          两个算子仓编译期各自 FetchContent 的依赖
├── assets/                可选：json/abseil/eigen/protobuf/makeself 归档
├── cann/                  可选：CANN 9.2.0 payload（仅从零构建镜像时需要）
└── prebuilt/              产物：压平后的 vendor 树 + wheel，供 --with-union-ops 用
```

`prebuilt/` 是产物，`ops-*/`、`cann*/`、`assets/` 体积大，都在 `.gitignore` 里；
入 git 的只有清单、脚本和本文件。

## 每一项是什么、为什么需要

| 目录 | 来源 | 用途 |
|---|---|---|
| `ops-transformer` | gitcode `HuangKun8682/ops-transformer_inedex_topk` @ `11661d45` | union 模式 GBSA 算子本体（`generic_block_sparse_attention`，maskType=3） |
| `ops-nn` | gitcode `bartonfang/ops-nn` @ `1c7fe8f1`（PR cann/ops-nn#12461） | `union_sparse_meta`（NPU 并集+掩码）及其 torch extension wheel |
| `cann-cmake` | gitcode `cann/cmake` tag `master-058` / `master-046` | 两个算子仓的 cmake 公共层。它们默认 `FetchContent` 去 gitcode 拉；发现 `<thirdparty>/cann-cmake/function/prepare.cmake` 就直接 `include`，这是**唯一的离线入口** |
| `opbase` | gitcode `cann/opbase` @ `0a83b915` / `40cc7bed` | 算子 host 框架。两个仓 pin 不同，且都会在自己的 cmake 里 `git checkout` 就地切换，所以每个仓要各给一份 |
| `ops-tensor` | gitcode `cann/ops-tensor` @ `99cc37f1` / `6d0ca6e1` | tensor_api 头文件；`include/tensor_api` 是子模块，两个 pin 要的子模块 commit 也不同（`2a241787` / `ad3d3bf0`） |
| `catlass` | gitcode `cann/catlass` @ `769cd40a`（v2.0.0） | GBSA kernel 用到的 tile 级 GEMM 模板 |
| `ops-test-kit` | gitcode `cann/ops-test-kit` @ `master` | UT kit。上游 pin 的是分支不是 commit，这里如实记录 |
| `assets` | 基础镜像 `/opt/step5/ops-thirdparty` | `json` / `abseil-cpp` / `eigen` / `protobuf` / `makeself` 归档。`build-union-ops.sh` 未指定 `--assets` 时直接从镜像里取，不必单独准备 |
| `cann` | `quay.io/ascend/cann:9.2.0-950-openeuler24.03-py3.12-devel` | CANN 9.2.0 工具链，仅 `ci/build.sh --full --reuse-base-assets=0` 时需要 |

**为什么必须是 fork**：上游/`stepx` 分支的 tiling 里写死了

```cpp
if (maskType_ != 1) {
    OP_LOGE(context->GetNodeName(), "Unsupported maskType=%ld, only maskType=1 is supported.", maskType_);
}
```

所以已发布的 `step5_ops_transformer` 收到 `maskType=3` 会直接报错，union 只能自己编译。
union fork 里这一段已经放开成 `0 / 1 / 3`。

## 编译

```bash
./build-union-ops.sh                              # 默认镜像、默认 assets、-j$(nproc)
./build-union-ops.sh --image <同一 CANN 9.2 镜像>  # 换工具链
./build-union-ops.sh --jobs 40 --clean            # 用完清掉 .stage/
```

脚本做的事，按顺序：

1. 给两个算子仓各准备一份依赖树 `.stage/{transformer,nn}/`：`opbase`、`ops-tensor`
   （含子模块）、`catlass`、`ops-test-kit` 用 `git clone --shared` 从本目录复制（秒级、不联外网），
   再 `git checkout` 到各自 pin；`cann-cmake` 用 `git archive` 取对应 tag。
2. 在容器里 `source cann-9.2.0/set_env.sh`，从镜像 `/opt/step5/ops-thirdparty`
   补齐 `json/abseil/eigen/protobuf/makeself`，用两个 shim 顶掉 `dos2unix` 和 `pigz`
   的前置检查，然后跑：

   ```bash
   # ops-transformer
   bash build.sh --ops=generic_block_sparse_attention,generic_block_sparse_attention_metadata \
        --soc=ascend950 --vendor_name=union_test --cann_3rd_lib_path=<thirdparty> -j40
   # ops-nn
   bash build.sh --pkg --experimental --ops=union_sparse_meta --soc=ascend950 \
        --vendor_name=custom_nn --cann_3rd_lib_path=<thirdparty> -j40
   bash build.sh --torch_extension --experimental --ops=union_sparse_meta --soc=ascend950 \
        --vendor_name=custom_nn --cann_3rd_lib_path=<thirdparty> -j40
   ```

3. 压平：`.run` 安装器会多套一层 `vendors/<name>`，python 侧发现不了，所以直接取
   CPack staging 目录 / `--install-path` 装到临时目录再把里层 `vendors/<name>` 提上来，
   结果放在 `prebuilt/{union_test_transformer,custom_nn_nn}/`，wheel 一并拷入。
4. 校验：两个 vendor 都必须有 `op_api/lib`（否则等于没装），`custom_nn_nn` 的
   `libcust_opapi.so` 必须导出 `UnionSparseMeta`。

已知坑（都已在脚本里处理）：

- CPack 最后一步要 makeself，离线拿不到时**只出 `.run.json` 不出 `.run`**；
  staging 载荷是完整的，直接用即可。
- ops-nn 的 `build.sh` **不认 `--incremental`**，失败只能整跑重来（USM 包小，几分钟）。
- `--vendor_name` 必须是 `custom_nn`（不是早期的 `custom`），否则 pip 包名/模块名对不上。
- 两个仓的 `build.sh` 前置检查要 `dos2unix` 和 `pigz>=2.4`，镜像里都没有。

实测（4×aarch64，-j40）：ops-transformer 约 7 分钟，ops-nn 两个包合计约 6 分钟，
全程零外网请求。

## 安装进镜像

`ci/build.sh --full --reuse-base-assets --with-union-ops` 会把 `prebuilt/` 放进构建上下文，
Dockerfile 用 `WITH_UNION_OPS=1` 把它 `cp -a` 到 `$ASCEND_OPP_PATH/vendors/` 并 pip 装 wheel，
装完用 `nm` 验 `UnionSparseMeta` 符号。`prebuilt/` 不完整时该开关直接报错，
而不是静默产出一个缺算子的镜像。

运行时要把两个 vendor 挂上 `ASCEND_CUSTOM_OPP_PATH`（冒号分隔）。跑完整服务时是
**三包串联**，`step5_ops_v2_transformer` 必须留在链上（它提供 IndexerRegionScore）：

```bash
export ASCEND_CUSTOM_OPP_PATH=$ASCEND_OPP_PATH/vendors/union_test_transformer:\
$ASCEND_OPP_PATH/vendors/custom_nn_nn:\
$ASCEND_OPP_PATH/vendors/step5_ops_v2_transformer
```

不设这个变量时 CANN 内置的 `libopapi.so` 会抢先，报错信息很有迷惑性
（`headDim currently only supports 128, but got 192`）。

两个包的 `libcust_opapi.so` 同名，但导出符号不相交
（`aclnnGenericBlockSparseAttention*` vs `aclnnUnionSparseMeta`），同进程共存实测无冲突。

## 离线/内网怎么把源码弄进来

构建网段常常连不上 gitcode。在一台能连的机器上 clone（**必须带 `.git`**，
`fetch.py` 靠它校验 ref，tar 导出的纯源码会被拒绝）：

```bash
git clone -b stepx_union --depth 100 https://gitcode.com/HuangKun8682/ops-transformer_inedex_topk.git
git clone -b feature/union-sparse-meta-a5 --depth 100 https://gitcode.com/bartonfang/ops-nn.git
git clone https://gitcode.com/cann/cmake.git cann-cmake
git clone https://gitcode.com/cann/opbase.git opbase
git clone https://gitcode.com/cann/ops-tensor.git ops-tensor
git clone https://gitcode.com/cann/catlass.git catlass
git clone https://gitcode.com/cann/ops-test-kit.git ops-test-kit
```

注意分支 HEAD 会往前走（`stepx_union` 现在已经不是 `11661d45`），
所以要在目标机 `git checkout` 到 manifest 里的 ref 再拷。
然后打包 → scp/rsync → 解压 → `fetch.py --from <name>=/path` 逐个落位校验：

```bash
python3 fetch.py --from ops-transformer=/tmp/ops-transformer \
                 --from ops-nn=/tmp/ops-nn
```

`--from` 只做拷贝和校验，不联网；缺 ref 的目录会被判 `failed`，不会"凑合用"。
