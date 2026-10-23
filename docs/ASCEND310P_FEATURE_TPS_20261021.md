# Ascend 310P C++ 继续实测（2026-10-21）

**2026-10-22 状态核查**：本轮已中断。用户要求停测时 SSH 连接超时，停止指令未能送达；此后板卡重启，旧控制器和模型进程均已消失。DFlash 最后日志停在预热，没有主跑结果；4B/9B 后续项目未启动。下文待完成队列是当时计划，不代表当前仍有进程运行。完整旧证据已归档到开发机 `results/feature-tps-20261022/feature-tps-20261021-evidence.tgz`，SHA256 为 `6dd81247a84c87b9af2438bc5b31112062d6a738b0cb0e3516ebfc7d880c02db`。

继续此前暂停的测试。Qwen3 MoE 以板上的 Qwen3-Coder-30B-A3B-Instruct 为目标，测试普通解码及其同名 DFlash；随后测试 Qwen3.5-4B/9B Dense。EAGLE3、Qwen3.5 MoE 仍跳过。旧结果和中断记录见 [2026-10-20 报告](ASCEND310P_FEATURE_TPS_20261020.md)，不覆盖旧失败。

## 实现与运行身份

本轮新增 `FLASHENGINE_TRACE_STARTUP=1` 可选 C++ 阶段日志，默认关闭，写 stderr；没有修改模型计算。日志分别标出 bundle 校验、权重装载、专家聚合/状态分配、执行计划、Graph 捕获、预热、主跑边界。主跑期间没有插入阶段日志，JSON 与原计时口径保持一致。

新二进制 SHA256 为 `1b8015328e3a8124e0f5c2ed0f2f4478ed2407853f8080033074a7a259aaf67c`，板端部署根为 `/data2/FlashEngine.cpp-deploy-b87b818-20261019`。源文件哈希、构建日志和旧二进制保存在其 `feature-tps-20261021/` 目录。板端编译成功；构建日志保留板端时钟落后导致的 clock-skew 警告，报告日期使用开发机的 Asia/Shanghai 日期。

正在运行的原生进程命令为 `flashengine-native benchmark ... --graph`。Graph 捕获前后均审计 `/proc/PID/exe` 与模块映射，二进制哈希相符，未加载 libpython/libtorch；链接的 ATB 为 cxx_abi_1。这不等于 MoE 已使用 ATB 优化：MoE 在当前分发中仍是 ACLNN ND 参考路径。

## MoE 普通解码的阶段定位

同一自然 prompt `Write a Python factorial function.`，thinking off，8 token 固定输出、ignore-EOS；warmup=1、repeats=2、batch=1，主跑总计 16 token。上下文 128、max_num_seqs=1、prefill 预算 32、partial-prefill=1，2 个可用 KV 页、页大小 128。沿用已冻结输入，不将该短负载代表长生成稳态。

| 阶段 | 进程内累计秒 | 本阶段秒 |
|---|---:|---:|
| bundle 验证完成 | 99.8563 | 99.8563 |
| tokenizer 验证完成 | 100.861 | 1.0047 |
| 开始装载权重 | 101.895 | 1.034 |
| 权重装载完成 | 354.828 | 252.933 |
| 专家聚合与状态完成 | 425.306 | 70.478 |
| target 执行计划完成 | 430.292 | 4.986 |
| target Graph 捕获完成 / benchmark 预热开始 | 490.010 | 59.718 |
| 预热完成 / 主跑开始 | 787.087 | 297.077 |
| 主跑结束 | 797.097 | 10.010 |

主跑之前的阶段均不计入 TPS。原生权重约 61GB，BF16 在 C++ 中转换为 FP16；不能把 30B 参数当成 30GB FP16 权重。本轮原生退出 0，主跑 16 token / 10.008951 秒 = **1.598569 TPS**；TTFT p50 为 **3381.660ms**，首 token 后 decode 为 **4.313689 TPS**，两者不能互换。两次重复分别约 5.005 与 5.004 秒，输出 token 相同，主跑 decode Graph replay 增加 14。首轮预热远慢于热态，当前证据只能记录这个差异，不能把未单独定位的差异全部归因于某一种编译或缓存行为。进程墙钟约 846.885 秒，包含环境准备、加载、预热、主跑与退出回收。

相同输入、相同可用 KV 容量与 Graph 模式的独立 Python 对照为 **13.529708 TPS**，TTFT p50 **263.034ms**，decode **21.371500 TPS**。两次请求的 8 个最终 token 和结束原因均严格相同（2/2）；没有执行完整中间 logits/逐轮 trace 比较，不能扩大为长输出验收。C++ 整体吞吐约为 Python 的 11.8%，decode 约为 20.2%，该性能明显尚未对齐。

源码差异与测量相符：C++ `Net::mlp` 是 FP32 softmax/topk 后逐专家 gather、BatchMatMul、累加，attention/prefill 也是 ND 单行参考执行；Python 使用融合 MoE gating、init-routing、NZ grouped matmul、SwiGLU 与 unpermute，prefill 整批计算。仅把参考执行放入 Graph 不会消除这些算子和布局差异。

Python 对照环境为 torch 2.9.0、torch-npu 2.9.0.post2.dev20260401、Transformers 4.57.6。其源码树 SHA256 为 `dd6b642dd9ba0e3745f92fff89d56828ea92cddf46d3e3531a55e069bf7738de`，与此前冻结输入的源码版本不同；变动涉及 ascend backend、block manager、engine、scheduler、quant、openai_server，故不复用旧 Python TPS。本轮没有修改这些 Python 推理源码。实际版本和控制器哈希位于 `python-runtime-identity.json`，差异清单位于 `python-source-changes.json`。

## 实际草稿配置预检

预检使用与导出入口相同的 normalize/configure 逻辑，仅读取配置，不执行 Python 推理、不修改检查点。

| 实际草稿目录（均在 /data） | 结果 |
|---|---|
| Qwen3-Coder-30B-A3B-DFlash | 配置通过；8 层、全词表、k=15；后续仍需权重导出与真机执行 |
| Qwen3-30B-A3B-Dflash | 失败：嵌套格式缺少入口所需 hidden_size；同时声明 draft_vocab_size=32000，当前 DFlash profile 不支持缩减词表 |
| Qwen3.5-4B-Dflash | 失败：滑窗 attention 不受当前原生 profile 支持 |
| Qwen3.5-9B-DFlash | 失败：滑窗 attention 不受当前原生 profile 支持 |

上述失败没有有效原生 TPS，不能删掉配置字段强行测试后称为原检查点通过。配置哈希与原始异常位于 `draft-preflight.json`。

## 待完成与证据

串行队列接下来执行 Coder-30B+DFlash 与 Qwen3.5 Dense；Python 仅在独立任务中作为输出与性能对照。Qwen3.5 两侧 KV 页大小明确设为 64，Python 配置适配使用修正了顶层 tie_word_embeddings 继承的 v2。此前的 9B 错误绑定权重基线不再作 oracle，Python Graph 的已有设备故障记录保留。

板端证据目录为 `feature-tps-20261021/`。`coder30-base-graph-v1/` 保存原生命令、进程状态、stdout/stderr、资源采样、运行模块审计；`followups.json` 保存后续队列状态。结果尚未齐全，不授予新增模型功能或性能验收资格。
