# 950DT 最新性能汇总（2026-10-23）

本轮 BS1–64 共21轮已完成，进程退出码0。输入131072 tokens、每请求输出2048 tokens、固定随机token输入，每档3次，未修改权重。测试后health200，8个DP均running=waiting=0，累计preemptions=0。

| 平台 / 精度 | 机器数 | 每机卡数 | 上下文长度 | Decode 总 BS | 每个 DP 的 BS | MTP 配置 | 接受率 | Prefill 吞吐（tps） | Prefill 达到 H200 基线 | Decode 总吞吐（tps） | Decode 达到 H200 基线 | ITL | 备注 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: | :---: | ---: | :---: | ---: | :--- |
| H200（基线-post21，用户提供） | 1 | 8 | 64k | 32 | 16 | MTP3 | 0.6 | 38,000 | 100% | 1,702 | 100% | 52.73ms | 精度/计量口径未另核验 |
| H200（基线-post27，用户提供） | 2 | 8 | 64k | 64 | 16 | MTP3 | 0.6 | 71,794 | 100% | 4,263 | 100% | 41.953ms | 精度/计量口径未另核验 |
| 华为 950DT / W8A8-MXFP8 | 4 | 8 | 128k | 1 | 0–1（未占满 DP8） | MTP3 | — | 8,977.20 | — | 33.03 | — | 30.66ms | P/D 同服务：DP8 TP4 EP32；3轮均值 |
| 华为 950DT / W8A8-MXFP8 | 4 | 8 | 128k | 2 | 0–1（未占满 DP8） | MTP3 | — | 16,352.63 | — | 65.32 | — | 31.26ms | P/D 同服务：DP8 TP4 EP32；3轮均值 |
| 华为 950DT / W8A8-MXFP8 | 4 | 8 | 128k | 4 | 0–1（未占满 DP8） | MTP3 | — | 27,005.55 | — | 128.07 | — | 31.84ms | P/D 同服务：DP8 TP4 EP32；3轮均值 |
| 华为 950DT / W8A8-MXFP8 | 4 | 8 | 128k | 8 | 1 | MTP3 | — | 41,851.69 | — | 312.10 | — | 31.26ms | P/D 同服务：DP8 TP4 EP32；3轮均值 |
| 华为 950DT / W8A8-MXFP8 | 4 | 8 | 128k | 16 | 2 | MTP3 | — | 43,588.47 | — | 1,287.02 | — | 33.57ms | P/D 同服务：DP8 TP4 EP32；3轮均值 |
| 华为 950DT / W8A8-MXFP8 | 4 | 8 | 128k | 32 | 4 | MTP3 | — | 45,916.26 | — | 2,892.01 | — | 36.08ms | P/D 同服务：DP8 TP4 EP32；3轮均值 |
| 华为 950DT / W8A8-MXFP8 | 4 | 8 | 128k | 64 | 8 | MTP3 | — | 46,851.22 | — | 5,627.83 | — | 41.15ms | P/D 同服务：DP8 TP4 EP32；3轮均值 |
| 华为 950DT / W8A8-MXFP8 | 4 | 8 | 128k | 128 | 16（目标） | MTP3 | — | — | — | — | — | — | 未执行；compact summary修正仍为未验证候选 |

## 口径

- 本轮为128k输入，用户H200基线为64k；缺少同上下文/同BS/同计量口径的H200结果，因此达标比例留空，不作公平性能比结论。
- Decode总BS为客户端发起并发；每DP列是均衡分配目标，不是已证明的全程同时驻留/稳态decode BS。BS1/2/4没有占满8DP。
- TPS/ITL为三轮算术均值。Decode TPS是客户端完整重叠窗口token吞吐，不是kernel-only/pure-decode吞吐。ITL是非空SSE事件间隔，多token chunk存在时不是逐token延迟。
- 各档MTP接受率未按iteration前后counter差值采集，不能填示例0.6，也不能从chunk数反推。
- 950DT精度是权重checkpoint W8A8-MXFP8；执行dtype BF16。benchmark JSON中system_config.dtype=bf16并不代表权重为BF16。
- 本轮P与D同一个DP8 TP4 EP32服务；不是历史P:PP4TP4/D:DP8TP4EP32分离部署。
- BS128未执行。现有summary4096-byte page padded至98304-byte，容量修正候选尚未通过硬件测试，不纳入本表。

## 身份与证据

- image：quay.io/ascend/vllm-ascend:v0.30.0rc1-a5-openeuler-step5-cann9.2.0-tailring-143f3649d8-final
- config：sha256:819e33aac7faff91480d4c6e5f861d1bc0c73c1c301fb40079eaad41f46f8e58
- vLLM ca5e43ddcd5b04c91e3de451702eecc1632faea8；vLLM-Ascend143f3649d8181edae635091c5fe809eb51ff992c。
- summary：ci/reports/benchmark-final-run1-summary.json
- post-state：ci/reports/benchmark-final-run1-post-state.txt
- 远端原始SSE/payload/iteration证据（约2.2GB）：CPU5 /data/chensiyu/step5-tailring-final-service-20261023/benchmark-final-run1

## 原始脚本统计口径复核

2026-10-23已用未修改的Downloads/benchmark_pd_tps (2).py统计函数，离线重放本轮21份iteration及请求原始时间戳。全部7档三轮平均Prefill TPS/Decode TPS与上表相同；Decode TPS相对差异0，ITL仅浮点末位差异。此次重放不是重新发请求运行原始client。

Decode窗口严格为 `(max(各请求首个实际token时间), min(各请求最后输出时间)]`，窗口内非首token的token_ids计数除以该窗口秒数。因此本批请求首token之前的TTFT/初始prefill时段没有加进Decode分母。但它是客户端E2E墙钟窗口，包含输出传输、客户端解析、调度/通信等等待；没有worker逐step证据证明为kernel-only或窗口内完全没有其他prefill/recompute工作。

此前Prefill Lat(s)177.8563是原脚本保留的最后一轮值，并非Decode分母。接受率仍未按档采集。

复核证据：ci/reports/benchmark-original-metric-replay.json。执行client安全副本SHA256 e74034901087d75f382cc26b0f6f6f752c87be46af69d0961da0052a7749ee11；原始脚本SHA256 d713f2f40016b3c55960a1403d5a202731065e9b6f18848440d337d85ca364a4。

BS64三轮实测 Decode：
- 第1轮：17.7262s，5740.20 tps。
- 第2轮：18.3781s，5477.60 tps。
- 第3轮：18.0235s，5665.70 tps。
