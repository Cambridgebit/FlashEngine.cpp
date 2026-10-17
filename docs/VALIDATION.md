# 本机验收记录（2026-10-16，candidate）

这是一份阶段记录，不是发布合格证。CUDA 尚未通过全部 P1–P4 门禁。下文 2026-10-16 为历史记录；2026-10-17 多后端/MoE 增量见文末，不能用新增合成样例覆盖既有失败。P5/P6 不得标为完成。

## 环境与可追溯工件

WSL2 Ubuntu 22.04 x86_64、RTX 4070 Laptop 8 GiB、GCC 11.4、CMake 4.4.2、CUDA 13.0.3 / nvcc 13.0.88、LibTorch 2.11.0+cu130、FlashInfer 0.6.14。完整来源/依赖限制见 `dependencies.lock.json` 与 `NOTICE.md`。

契约版本 0.1.0，发布 tar.gz SHA256：`8697651192804ae584308eb53106af43a0389df5e1340fbc3e1b0ae0a7811181`。快照位于 `vendor/contracts/`，主机测试逐文件核验 `contracts.lock`。当前 oracle 来自 Python HEAD `17f1a4b792fecac48d6aaa7962533dd52f595368` 及未提交增量；新版导出器额外记录实际源码树哈希和 dirty 状态，不能仅用该 HEAD 复现所有新增工具。

本地完整工件保存在 Python 父仓的 `deployment_output/`，它不被 Git 跟踪。报告含执行命令、binary/oracle/bundle SHA256、完整结果、stderr 和差异位置；二进制更新后必须新建报告，不能覆盖旧报告。

## 已观察到的结果

- Python 纯主机单元测试 8/8 通过，覆盖权重完整性、shape/截断、重复 JSON 键、非标准常量、支持面拒绝及契约工件可重复/不可覆盖。
- C++ CTest 两组通过：168 项主机断言（含契约快照、所有 k=1..15 接受前缀长度、小 KV 池强制抢占恢复、上下文硬停、取消、mmap 边界）及引擎并发/幂等取消/独立子进程 fail-fast=70。
- ASan/UBSan 主机构建的两组测试通过；这不代表 CUDA 内核已通过内存检查。
- Qwen3-0.6B bundle 的 310 个目标张量以及 1.7B 合成 DFlash bundle 的 310+58 个张量通过原生校验。Unicode tokenizer 扩展差分 205/205，bundle 内 chat template 字节/token golden 通过。
- 0.6B 普通 CUDA eager 与 Graph 的三组输出相同；Graph 实际 replay 45 次。对 Python flash-attn/FlashInfer oracle 都是中文、代码两组各 16 token 一致，英文输出共同前缀 12 token 后分叉。首个分歧处 Python top-2 margin=0，原生 margin=0.0625。报告仍为 `passed=false`，没有豁免。
- 1.7B + 合成 DFlash k=2 的 eager 和 Graph 最终 token 都与 Python 三组各 16 token 一致，Graph 的 draft/verify replay 各 45 次。逐轮 trace 未通过：部分提案/verify ID 不一致，包含非零 margin 样例；尚未逐层定位，不能统一归因于平票。所有合成草稿样例的接受数为零，不覆盖真实加速或非零接受状态。
- 原生服务 27 项检查通过：鉴权/非法请求、models、普通与 chat 的 SSE/非流式一致、跨 token 文本 stop、usage、并发计数、断连取消、KV 归零、metrics/stats 同源及 SIGTERM 正常退出。观测到 running=4、waiting=4、active=4；Graph 有非零回放。早先两次停机失败报告保留，后续 v3 修复通过。最终观测语义修复后的 max_num_seqs=7 回归也通过，running/active 峰值均为 7，图桶 `[1,2,4,7]`，decode replay=91；TTFT 主机计数按 Python 口径不含首次调度前排队，queue 单独记录。
- ELF 动态依赖检查未发现 libpython、libtorch_python 或 Rust 共享库；构建证据包含 native 源码/头文件和实际动态库哈希。这不是全部供应链/静态链接审计，也不是正式离线依赖包。
- 继续扩展 batch=8 时曾复现一次在线 27 项通过但退出段错误，旧报告 `native-server-batch8.json` 保留。已把 acceptor 与信号关闭绑定同一 strand，并先 stop/join 引擎再销毁网络上下文。修复后的 batch=8 服务 27 项通过（running/active=8、decode replay=63），真实 CUDA 生成中带 8 个半包连接停机 3/3 退出 0；仅测试主机模型的同一 Beast 路径在 ASan 下重复停机 5/5 通过。主机模型不是设备推理验收，不进入部署程序。
- 大词表解析改用标准映射，去除了 ordered_json 线性查找造成的二次复杂度；仍拒绝重复键，优化后 Unicode 差分 205/205。早先 sanitizer 启动超时报告保留，不作为停机通过证据。

## 关键报告

以下是固定历史报告的 SHA256；每份报告内部另含其测试二进制哈希。

| 工件 | SHA256 |
|---|---|
| `native-server-graph-v3.json` | `289397f83fb69c5e6cdd6f98a67991b30d594338cfe7801009efef135dfb96c7` |
| `native-server-tail7.json` | `439a6ea5a41be2d601116c89670076d3dfc3dec82605a4a7524caf225e49f799` |
| `native-server-batch8-final.json` | `10dcc37edd9dae2ce4ab1cd59a1bdf071cabcb59e2e90ed5e53430936bd794d2` |
| `native-shutdown-cuda.json` | `3e97a21ae71ea788cb9ccaae7194066ab6a0613807177c4c031f5e38ac4a6170` |
| `native-dflash-k2-logits.json` | `6880c378907b439df7e2e469dad35784f191819f6b1ed3ff7144f2135df23ade` |
| `native-fa-eager-logits.json` | `c056c93d5dd99fe0c2ee63562fe4f74ce28e73f6dfe5c0b72d58b3ea34a7b545` |
| `native-fi-eager-logits.json` | `56add858f83d86db2092833629f281a07aac3ef60036a4fa930e9cbec662800f` |

## 仍需完成的阶段出口

P0 已有双仓/契约/打包与 host 构建测试；完整依赖 artifact pin、golden 发布锁和自动化 CI 仍待补齐。

P1 必须完成逐层 logits 差分、明确平票见证策略和更多无平票样例，才可宣布通过。CUDA 目前只编译 head_dim=128。

P2/P3 已观察到并发、取消、chunked prefill、非标准尾桶、图回放及有请求停机，仍缺全量 batch 1/2/4/8 的逐形状数值差分、桶复用/请求重排的严格 oracle 对表、端到端抢占、慢客户端/队列满及 CUDA 内存错误检查矩阵。

P4 必须先收敛真实草稿 trace，再加入受控 m=0..k 的设备接受测试、多请求状态生命周期、EOS/max_tokens/max_model_len 轮内截断和稳态图命中门禁。合成草稿不证明收益。

P5 需要实际 CANN/ATB/ACLNN 头文件、目标容器和 AArch64+310P 板卡；不得通过 Python/TorchNPU 绕过纯 C++ 约束，不能以 CUDA 或仅交叉编译替代真机验收。

P6 尚缺热态交替 A/B、长稳/故障报告、完整离线依赖包、OCI 镜像、release manifest 与 supervisor 运维验证。当前没有性能合格结论；TPS >=95%、p50 TTFT <=110% 的发布门禁保持不变。

## 2026-10-17：MoE / 多后端源码与部署增量

契约升级至 0.2.0，bundle 格式 1.1.0；保留旧快照和 0.1 dense bundle 读取兼容。新契约归档 SHA256：`8245808a803936ea541fa81343cf05cb894f55817e57f9cd4c3e5534ae736d59`，快照为 `vendor/contracts-0.2.0`。模型 profile 增加 MoE sparse-step/dense 层、top-k 归一化，以及逐专家/融合 3D 装载计划。

本轮已实测：

- Python 纯主机测试 12/12；C++ 主机断言 184 项及引擎生命周期测试通过。
- 最新 ASan/UBSan 主机构建两组 CTest 通过；不把它当作设备内存检查。
- 真实 Qwen3-0.6B **FP16** eager 与 Graph 对 Python 三组各 16 token 严格一致，Graph replay=45；同时导出 top-8 logits 供诊断，尚未宣称完整逐层 logits 门通过。该结果不豁免旧 BF16 分歧。
- CUDA 新增真实设备接受测试：batch=1/2/4/8、k=1..15、每个接受前缀长度，eager 与 Graph 共 4050 项检查；Graph 改变输入后实际 replay 540 次，输出区前后 sentinel 未被越界改写。它验证接受算子，不冒充全部模型状态矩阵。
- 高 margin 合成 MoE（4 层、4 专家、top-2、两层 dense / 两层 sparse）普通 Graph 与 Python 三组各 16 token 相同；DFlash k=2 三组各 5 轮逐轮严格一致、均满接受，draft/verify Graph 各 15 次。逐专家与融合 3D 两种装载路径均有报告。
- 融合专家 MoE+DFlash HTTP/SSE 27 项通过，配置 `max_num_seqs=7`、图桶 `[1,2,4,7]`、prefill 预算 8 token、KV 页大小 16 / 页数 16；观察到真实抢占 1 次、running/active 各自峰值 7、取消后 KV 归零，draft/verify Graph 各回放 36 次，退出 0。
- `cuda-reference` 的纯 C++ ATen 算子在 NVIDIA 上通过同样的 4050 项检查 / 540 次回放，以及融合专家 DFlash 严格三组差分、draft/verify 各 15 次。**这些结果不代表 CoreX 已编译或真机通过。**
- 新构建证据读取二进制 `capabilities`，核验动态依赖无 libpython/libtorch_python/librust，记录当前契约快照、ELF 主机架构及依赖哈希；仍不是完整供应链审计。
- 算子级 oracle 覆盖 FP16/BF16、宽度 128/2048、RMSNorm/add-RMSNorm/残差/SwiGLU，共 16 组、139264 个元素，原生 CUDA 与实际 Python 编译结果逐元素位等。实测 Python 编译优化消除了源码表面的中间 dtype 回投；强行补一次回投反而会偏离 oracle，因此没有修改原生算子的舍入顺序。

合成模型通过 `scripts.create_native_moe_fixture` 创建，以高 margin 常量 token=42 强制满接受。它不是训练模型，不能证明真实 MoE 模型质量、数值边界、接受率或加速收益。此前 0.6B BF16 和 1.7B 合成草稿逐轮差分失败报告继续有效。

进一步补测 1.7B+合成 DFlash FP16：eager/Graph 两份严格报告仍失败。英文共同前缀 5 token，第 4 轮目标 verify 的 Python 两候选平票，原生出现差异；中文最终 token 相同但未提交 verify 行不同，代码样例通过。Graph 的 draft/verify 各 replay=45。不能把这些结果扩大解释为全部差异均已定位或可豁免。报告器新增投机 logits 按轮/行处理、超时和诊断失败现场归档，避免把嵌套行误当全局 token 的 margin。

新增固定工件：

| 工件 | SHA256 |
|---|---|
| `moe-fused-dflash-cuda-graph-v1.json` | `950311b3cf5c1fa2f6de9e9d71aabd0e0022baa4b4f1777e187c9088a9d68382` |
| `moe-fused-dflash-server-pressure-v1.json` | `40c24da2aa7e38e12f544009d975f317a32a81fd788907e0f3fc124fcc2fd284` |
| `native-build-multibackend-v2.json` | `86da066247dcebec774f1e76d847169746274ab7c51930ca792d4fc844a59bce` |
| `native-build-multibackend-v3.json`（含最终算子测试源码） | `52103c8cc3e219325296bc6cdbb0b5e6eaffa3e6f851c94256de849b9967c7f4` |
| `native-ops-cuda-v1.json` | `85dea789ccfc8346ab52d81ef4b8ce5568429b694b97c1cd5e49fbc4a6f825bb` |
| `native-ops-reference-v1.json`（NVIDIA 上的参考算子） | `561905602f45b192d500ef4e5d89ed636b4f81866b522cc4f73b72adcfc7854a` |
| `moe-fused-dflash-reference-graph-v1.json` | `8c2a23805ad9cb697baf38ce98820a32b1f49f084c4ddb0616ac50b5c59c1b4c` |
| `qwen3-0.6b-fp16-native-eager-v1.json` | `ecc0241d1a8de251f88480fcd44aa1f07c74b1b98b2a36001ee8aa37651196e7` |
| `qwen3-0.6b-fp16-native-graph-v1.json` | `c09ff6a260de3e74d7239c6f15ccfb38ccf1e68b50e0a1d7bbfc85af7fbf3236` |
| `qwen3-1.7b-synth-fp16-native-eager-v1.json`（失败） | `5467e20d5d32d39cb832aa3884fdc64aedb0ffd391f3191e09a8aa6198201bea` |
| `qwen3-1.7b-synth-fp16-native-graph-v1.json`（失败） | `b6d00f00f40efe8d354df7cf2d97d1bd473a2cb4d70a0483eb597c91d47b8282` |

CoreX 新增厂商 LibTorch C++ / ATen / Graph 的构建入口；310P 新增纯 AscendCL / ACLNN / ACL Graph 的 dense/MoE/DFlash 参考源码。二者无 SDK，**未编译、未上板**：本机配置分别在厂商 LibTorch/SDK 门和 `acl/acl.h` 门明确失败，没有把这类失败写成构建通过。310P 当前为 ND 参考 attention，ATB/NZ 高性能路径未实现；CoreX 也尚无原生 ixformer 高性能接入。

部署辅助包含 `docs/DEPLOYMENT.md` 的 x86_64/WSL2/AArch64 命令、SDK/ABI 检查、KV/专家容量说明、HTTP/SSE 与验收流程，以及 `packaging/` 中的 systemd 和 digest 固定基础镜像模板。模板未完成镜像发布或 supervisor 真机验收。所有后端仍为 candidate，没有 production 发布资格。

## 2026-10-17：严格 CUDA 矩阵与 Python/C++ benchmark

新增原生 `benchmark` 命令及 Python 冻结输入/ABBA/HTTP/审计工具。运行口径见 [基准说明](BENCHMARK.md)，完整命令和实测表见对应 Python 仓 `docs/BENCHMARK_NATIVE.md`。两种吞吐单独报告，合成夹具 TPS 不作训练模型性能声明。

- Python host unittest 21/21，C++ host CTest 2/2、ASan/UBSan CTest 2/2、CUDA CTest 3/3。
- 真实 Qwen3-0.6B FP16、8 条×16 token、batch 1/2/4/8、eager/Graph：64 对请求最终 token 严格相同，各实现自身 eager/Graph 也相同；配置、结果哈希、测量期 Graph 审计通过。
- 扩展至 128 token 的 8 条串行基准，仅 7/8 Python/C++ 严格相同。teacher forcing 在同一前缀确认第 22 个 token 分歧：Python token=1447，top-2=[23.6875,23.671875]；原生 token=13，top-2=[23.671875,23.671875]。原生 eager/Graph 的全部 teacher token/样本相同；Python margin 非零，不豁免。完整逐层定位仍待完成。
- 附加 Python FlashInfer 对照：同一 teacher 前缀的 128 个 argmax 与原生全部相同，但仅 1/128 个 top-8 样本位等；自由生成同一 8 条×128 token、batch=1，在 eager/Graph 均 8/8 严格相同，各自两模式也相同，配置/哈希/Graph 审计通过。它不是主性能基准，不覆盖默认 FlashAttention 的失败，也不扩展到 BF16/DFlash/其他批次。
- 高 margin 融合专家 MoE+DFlash k=2：9 种长度、输出 17、上下文 64、预算 8、KV 4×256、batch 1/2/4/8、eager/Graph 的 16 个 worker，最终 token 和每轮 draft/verify/接受/提交/上下文严格一致；包括真实抢占、组尾、chunked prefill 和上下文尾部单 token 路径。
- 三组 batch=8 eager/Graph 边界夹具，各 9 请求：部分接受后 bonus=EOS → `[42,43,EOS]`；已接受提案中 EOS → `[43,EOS]`；max_tokens=2 在原始 `[43,EOS]` 提交中截断 → `[42,43]` / length。两端 trace/完成原因严格相同，并直接断言上述预期；原生测量期 draft/verify Graph 分别非零。
- Compute Sanitizer memcheck：接受算子 4050 项/540 replay、MoE+DFlash batch=8 Graph（含抢占）、真实 0.6B batch=8 Graph 的 16-token 路径，三条命令均 `ERROR SUMMARY: 0 errors`。未做全量形状/racecheck/长稳，不能扩大解释。
- 两轮 ABBA 的 Graph 串行中位吞吐 Python 153.12 / C++ 178.48 tok/s；eager 为 31.46 / 167.81。HTTP Graph 并发 1/4/8 分别 Python 147.58/361.05/616.24，C++ 165.59/607.69/1025.98 tok/s。仅是同机诊断性能，严格失败使发布门保持关闭。
- HTTP 首次发现 Python Completions SSE 文本误放 delta.text，旧 v1 缺测 TTFT 无效且保留。修复为 choices[].text 后独立 v2 重跑：96 个测量请求，usage 与服务器计数一致；配置/哈希/TTFT/Graph/KV 审计通过。Python 并发 8 有接入与组批波动，保留两轮，不挑最快值。原生停机退出 0；Python -15 不作为优雅停机通过。

固定工件 SHA256（均在 Python 父仓忽略的 `deployment_output/`，须独立归档）：

| 工件 | SHA256 |
|---|---|
| `strict-qwen06-fp16-batch-v1/summary.json` | `21739ea803504cccfc34e0fe6a381bdfc791360f1d37ab0411ac223dec407974` |
| `strict-qwen06-fp16-flashinfer-v1/summary.json` | `7eee9b635516b54cd9dcaeaeec9e08938c461058c2ed221e15393e0d244e0f75` |
| `strict-moe-matrix-v2/summary.json` | `a54282549122dcca7d004cb0b81df5d726493266a82f4918985f4295d750c355` |
| `bench-qwen06-fp16-serial-v1/summary.json` | `b3bb1b6d154e894f13ff0f506aa551dc98eacc279d9cd0a25bd255a19e149ec5` |
| `bench-qwen06-fp16-http-v2/summary.json` | `19ed7a4f1ee10d7db842bacd1367183123ee141f130a0375363645e0d769183b` |
| `cuda-acceptance-memcheck-v1.json` | `10be2a114319376e80505bf70421f61f2653390ea0aa759bc675ec229d97f1ae` |
| `cuda-moe-dflash-memcheck-v1.json` | `c2a27370c4c2f77709eb6e1e9ba714fd77ca189480c44f7f18c098db9358cfb9` |
| `cuda-qwen06-graph-memcheck-v1.json` | `b150af31a36f7734ae2fd324fff64f233b21ad062eb03d9fceb28ed053520399` |
| `native-build-cuda-benchmark-v1.json` | `8284690313fd829a346f592558c13bbe0555ce4b3c0e0f65228fb788f7d6fad7` |

短输出与合成边界通过不覆盖真实长输出、BF16 和训练草稿 trace 的失败。全部后端仍 candidate。
