# 版本化共享构建缓存

`ci/cache/` **不整体排除 Git**。版本说明、输入清单和 LFS 规则随仓库提交；
大归档由 **Git LFS** 存储，Git 提交的是指向对应内容的 pointer。
其他人拉取对应提交及 LFS 对象后，取得同一份可校验的构建输入。

## 目录布局

```text
ci/cache/
├── README.md                         # 普通 Git 文件
├── manifest.json                     # 普通 Git 文件：版本、大小、SHA256
└── official-v030rc1-cann9.2.0-317c7fb99e/
    ├── vllm.tar.gz                    # Git LFS
    ├── vllm-ascend.tar.gz             # Git LFS
    ├── ascend-catlass.tar.gz          # Git LFS
    ├── custom-assets.tar.gz           # Git LFS
    ├── ops-git-metadata.tar.gz        # Git LFS
    └── cann92-release/
        └── cann-9.2.0-950-openeuler24.03-py3.12-devel-ascend.tar.gz
```

最后一个文件也是 Git LFS；是否已经可用以实际文件和校验结果为准。
不能仅因目录或清单存在，就认定 4.8GB CANN 文件已经下载或上传。
默认 CI 输入使用此版本子目录；`--inputs-dir` 可显式切换到其他已校验目录。
只忽略 `.part`、`.tmp` 和 `cache-status.local.json`，不忽略有效归档。

## 为什么不用普通 Git 保存大文件

五个现有源码/依赖归档合计 321898766 bytes，另有 4806522202 bytes 的 CANN layer。
普通 Git 会把所有历史版本永久放入仓库对象，clone 和历史增长成本很高；
且本仓库的 GitHub remote 有单文件大小限制，不能直接提交两份超过 100MiB 的归档。
本设计仅准备 LFS 跟踪规则，不意味着任何 remote 已确认开放 LFS 或具备容量。

内部 GitLab 更适合作为默认发布目的地，但提交前必须确认：

1. 项目 LFS 已启用，所有构建者都有下载权限；
2. 存储额度及单对象限制允许 4806522202 bytes 的对象；
3. 发布者能实际上传，另一台机器能在干净 checkout 中拉取并通过 SHA256 检查；
4. 不同 remote 的 LFS 对象不会自动同步。仅向内部 remote 发布，GitHub 用户不能
   假定能够获取这些对象；分享给 GitHub 使用者时要另行安排许可与存储。

如果服务端不能接受整个 CANN 对象，优先保留带版本的官方镜像下载作为 fallback，
不要静默退回普通 Git 提交 4.8GB 文件，也不要生成未配置存储服务的虚假下载 URL。

## 把现有归档加入缓存

从实际已有下载目录导入，不重新下载、不改变源码、原文件不删除：

```bash
python3 ci/scripts/seed-shared-cache.py \
  --source /data/chensiyu/hw_project/step4/950_build/upgrade-v030-20261022
bash ci/build.sh --check-only
```

导入前后均检查 SHA256 和大小；有效缓存直接复用，损坏缓存拒绝覆盖。
脚本兼容旧 CANN 输入名，只在导入时把它保存为新版可读名称。
这减少重复下载，**不代表 native 编译可以跳过**。

## CANN 的版本可读下载入口

来源为官方镜像 **`quay.io/ascend/cann:9.2.0-950-openeuler24.03-py3.12-devel`**：

```bash
# 只核验版本标签和 ARM64 manifest，不下载大文件。
python3 ci/scripts/fetch-cann92-payload.py \
  --image quay.io/ascend/cann:9.2.0-950-openeuler24.03-py3.12-devel --resolve-only

# 下载到默认共享缓存位置；有效现有文件命中缓存，不重复下载。
python3 ci/scripts/fetch-cann92-payload.py \
  --image quay.io/ascend/cann:9.2.0-950-openeuler24.03-py3.12-devel
```

脚本从带版本的官方 tag 选择 linux/arm64，再验证锁定的 manifest 和 payload；
registry 底层仍按 digest 寻址，这是内容校验机制，不作为用户面对的文件名或入口。
这份 tar.gz 是从官方 OCI 镜像取出的 Ascend layer，**不是官方另发的同名安装包**。
其他依赖保持 session 相同内容；Dockerfile 仅改 CANN 文件名及来源注释。

## 发布者操作（需确认 remote LFS 可用后执行）

以下仅是发布说明，维护脚本不会替用户执行 commit/push：

```bash
git lfs install --local
git check-attr filter -- ci/cache/official-v030rc1-cann9.2.0-317c7fb99e/vllm.tar.gz
# filter 应为 lfs；不能直接 git add 后让大文件进入普通 Git。
git add ci/.gitattributes ci/cache ci/pins/official-v030-cann92.json
git lfs status
git lfs ls-files
# 先审核暂存 diff，确认所有缓存归档暂存成 pointer，再做签名提交。
git commit -s -m "ci: share versioned Step5 build cache through Git LFS"
git push internal HEAD
```

push 依赖 Git LFS pre-push hook 上传对象；仅上传 Git pointer 不足以共享真实文件。
如果只改了工作区还没有 commit/push，其他人仍然无法通过 clone 取得新增缓存。
真实文件、pointer 和 manifest 三者必须保持同一 SHA256/大小。

## 使用者操作

```bash
git lfs install --local
git lfs pull internal --include='ci/cache/**'
bash ci/build.sh --check-only
```

remote 名称按自己的 clone 配置替换，`internal` 不是通用要求。
CI 检查会把尚未取回的 LFS pointer 明确列为 `unhydrated_lfs_inputs`，
绝不把一个 130 多字节的 pointer 当作 tar.gz 解压。
可访问官方 Quay 的使用者也能用上面的版本化 CANN 下载脚本填充该项。

编译生成的 wheel、模型、容器镜像、密钥、日志和 `csrc/build_cache` 不属于本次
共享输入缓存，不能无选择地提交到此目录。
