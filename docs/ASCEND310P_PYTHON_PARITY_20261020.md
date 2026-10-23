# Ascend 310P：Qwen3-8B / DFlash 对齐记录

范围是 `192.168.16.118`、容器 `vllm018` 内的 Qwen3 dense FP16、TP=1、greedy，以及 `/data/Qwen3-8B-Dflash` 的 k=15 草稿。其他模型、量化和后端不随本报告获得验收资格。当前仍为 candidate；最终输出一致、接受长度一致与完整逐轮 trace 一致分别记录，不能互相替代。

## 最终 Graph 与严格差分

最终实现 `native-v8`：4 条 coding 请求连续两遍，共 218 个实际输出 token。8/8 请求最终输出、接受长度及完整逐轮 draft / verify / committed / context trace 与冻结 Python oracle 严格一致，共 26 轮。原先出现差异的 coding-1 还单独执行带 top-8 logits 诊断的 eager replay，退出 0、全部轮次一致，未重新生成 oracle 或放宽平票。

| 测量阶段 | 输出 token | 吞吐 token/s | TTFT p50 ms |
|---|---:|---:|---:|
| Python 首遍（导出 trace） | 109 | 45.8351 | 165.3616 |
| Python 后续热态（关闭 trace 读回） | 109 | 46.2914 | 160.1068 |
| 原生首遍（包含新长度执行计划准备） | 109 | 32.7973 | 552.3113 |
| 原生第二遍（执行计划已缓存） | 109 | 55.3513 | 105.3394 |
| 原生两遍整体，单外层墙钟 | 218 | 41.1827 | 见逐请求 JSON |

单遍原生数值按该遍四个串行请求的 group wall 求和；整体数值使用 benchmark 原始外层墙钟，包含两遍，不用热态值替换。Python 两次测量中只有首遍安装额外的逐轮诊断读回，后续性能测量关闭该钩子。该短负载显示真实加速，也显示首次新长度的准备开销尚未达到 Python 水平；不是热态交替 ABBA、长稳或正式发布性能门禁的完整验收。

测量期间整轮 Graph 实际 replay 增加 26 次，draft / verify 计数各增加 26，verify 行增加 416，capture failure 为 0；不能把 warmup 的回放算进主跑证据。

最终二进制 SHA256：`aabf6e2aa8d3184ddf75c21c7a141a44bcce8150b7cbaa8fd3016958d72259a5`。对应 `ascend_qwen3.hpp`：`afb4f70daea7c1613f57696ea4d0243afcb77c27c5cf87918a5823fe68404c72`；`ascend_ops.hpp`：`e5fd0fde4294f613c2a7e2bb9a71251be57f0b570b96ede0c6e3ee673591c489`。全部编译输入哈希位于 `native-v8/identity.json`，不能只用未提交增量之前的 Git HEAD 识别本实现。

严格一致指本次测试的 token 与逐轮状态，不代表全部浮点分数逐位相同。单独 eager 诊断保存 217 行 top-8：216 行的候选集合相同，43 行的候选及分数完全相同，共同候选分数的最大绝对差为 0.1171875。所有实际 greedy 选择和完整逐轮 trace 一致；分数差异仍保留，不用阈值抹去，也不能推断任意新输入均不会出现 FP16 边界翻转。

## 真实设备回归范围

- NPU 算子回归 13 类通过：Python RoPE 舍入边界/布局、连续单行视图字节数与偏移、8B 实际 Q/K 头形状、非方阵 NZ Linear、BF16/F32 权重转换、Q/K 输入别名与动态 Graph、不同头数 RoPE、因果 Flash、非等长双序列隔离、设备零/部分/全接受、SplitFuse 动态上下文、融合 Norm/SwiGLU、动态 KV 槽位。
- 最终 `boundary-v4` 使用真实模型权重及构造的边界 token 输入：6 条请求、batch=3、`max_num_seqs=4`，长度 244 / 180 / 27 / 244 / 180 / 370，输出上限 48 / 48 / 48 / 48 / 48 / 14，全部 ignore-EOS。与独立 Python 运行逐 token 和结束原因 6/6 相同；此项不宣称自然语料质量或完整逐轮 trace 对齐。
- 边界配置 `max_model_len=384`、prefill 预算 96、partial-prefill=3、12 可用页加 dummy。覆盖 1/2/4 桶、三请求到四请求桶的 padding、跨 KV 页、Prefix 复用、不同长度查询与上下文尾部。含一次三请求 warmup 的累计调度计数为：Prefix 请求 9、命中 6、命中块 6、chunked prefill 步数 9、prefix eviction 1；结束后 running / waiting / active / used_blocks 均为 0。
- prefill 执行计划 LRU 保持 4 项，实际驱逐 4 次；没有把逐请求长度的设备缓冲无限保留。
- 最终 `server-v4`：普通 completion 与 SSE 文本一致、3 请求并发、断连取消计数增加、running / waiting / KV 归零、metrics 可读、SIGTERM 退出 0。测试端口仅绑定容器内 `127.0.0.1:18018`，完成后已停止服务。
- 服务运行时保存 `/proc/PID/maps`，确认实际加载 `cxx_abi_1` 的 ATB，未加载 `libtorch` 或 `libpython`。
- 最终重建主机测试二进制后，CTest `host-contracts`、`engine-lifecycle`、`ascend-kv-dynamic-slots` 共 3/3 通过；设备组包含上述 13 类回归。

## 普通 8B 对照与结束状态

额外导出不带草稿的独立 bundle，未改写 DFlash bundle。两个 bundle 的目标权重文件哈希相同；普通路径用同一四条预分词请求、相同可用 KV 容量和一次 warmup。4/4 最终 token 及结束原因与 Python 相同，主跑 decode Graph 实际 replay 增加 105 次。

| 普通 Qwen3-8B 指标 | Python | 原生 |
|---|---:|---:|
| 含 prefill 的整体吞吐 token/s | 10.5461 | 9.4827 |
| TTFT p50 ms | 173.2447 | 532.5954 |
| 首 token 后的 decode token/s | 10.8886 | 10.7354 |

decode 口径为 `Σ(output_tokens - 1) / Σ(e2e_seconds - ttft_seconds)`，两侧均为 105 个 decode token，不能替换上表整体吞吐。普通目标模型的 decode 速度约为 Python 的 98.6%；整体与 TTFT 的差距主要仍在新 prompt 长度的一阶段执行计划准备。该开销保留为当前限制。

最终审计 `final-audit.json` 逐项核对：二进制/源码哈希未漂移，8 个 coding 主跑请求的 26 轮严格 trace 通过，6 个边界请求与 4 个普通请求对照通过，服务与 CTest 通过。结束后未遗留本轮模型或服务进程；`npu-smi info` 返回 0、设备 Health=OK、AICore=0%。驱动对该板的 Memory-Usage 显示 NA，不能将其写成显存使用 0 MB。

完整证据已取回开发机：`results/ascend-python-parity-20261019/evidence/parity-20261019/`。其中 `native-v8/graph.json` 是原始两遍总墙钟，`native-v8/comparison.json` 含分遍统计，`diagnostic-v2/` 保存修复后 eager 诊断，`boundary-v4/`、`server-v4/`、`base-v1/` 分别保存边界、服务和普通路径结果。`parity-evidence-final.tgz` 同时保留失败尝试及构建历史；旧失败文件没有改写为通过。

## 执行实现

| 环节 | 旧原生参考路径 | 本轮 Qwen3 路径 |
|---|---|---|
| 线性层 | 独立 Q/K/V、gate/up，ND 权重 | 打包 QKV、gate/up，NZ 权重 |
| Attention / KV | ND gather 参考 attention | ATB Flash / Paged / SplitFuse，NZ KV |
| target verify | k+1 次逐行执行 | 一次批量验证 k+1 行 |
| DFlash 上下文 | 辅助隐藏状态逐行下载 | FC、hidden norm、跨层 K/V 投影全部保留在设备 |
| 接受与回滚 | 主机接受，逐行重建 | 设备首不匹配位置、接受掩码、dummy 槽位 |
| Graph | 草稿图 + 单行 target 图 | 草稿、target verify、接受、上下文投影组成整轮 ACL Graph |
| 读回 | 多次同步及 aux 下载 | 每轮一次 token / trace 缓冲读回；`--logits` 诊断除外 |
| 冷 prefill 补齐 | 旧路径逐 token；中间版本整层补齐 | 仅 Flash Attention 内部补齐到 16，其他算子保留真实行数 |
| 上下文收尾 | 中间版本整批回退 | 正常请求继续投机；剩余窗口不足一轮的请求按实际批大小 eager 收尾 |
| 缓存生命周期 | 参考实现的主机附属状态 | draft 与 target 使用相同物理页；prefill 执行计划 LRU 上限 4 |

RoPE 使用公开 `aclnnApplyRotaryPosEmbV2`、BSND、half 模式。接口映射参照 [Ascend op-plugin 的实现](https://raw.githubusercontent.com/Ascend/op-plugin/master/op_plugin/ops/opapi/ApplyRotaryPosEmbNpuOpApi.cpp)，与板端 Python 当前选择一致。上下文投影的 Q/K 允许共享输入，原生实现先复制到独立输出，避免原地旋转污染源张量。

RoPE 常量也按 Python 的顺序在 NPU FP32 上计算：ScalarTensor Pow、倒数、位置外积、Cos/Sin，再转 FP16。不能用主机 `libm` 的数学近似值替代；独立探针对前 256 个位置的 32,768 个 FP16 值逐位比较，CANN 路径差异为 0，原先主机路径有 120 个不同值。

`--logits` 在这条优化路径上输出真正的 draft / verify top-8，不再静默切换到旧 ND 参考模型。诊断读回不进入常规 benchmark。

## 固定输入与环境

- HiSilicon AArch64 / Ascend 310P1；CANN 9.0.T3；ATB 9.0.0，C++11 ABI=1。
- Python 使用板上已有 `/data2/FlashEngine`，Git `ce94022b3685bfe4de29c7db6ca5bef6d529469f`。核心 runner/backend/model/config 的文件哈希已与开发机核对；未为对照修改 Python 引擎。
- Python 实际运行时为 torch `2.9.0+cpu`、torch-npu `2.9.0.post2.dev20260401`；执行为 NPU FP16。其 runtime 报告 `flash=legacy`、`splitfuse=legacy`、NZ 权重和 `GEMM=torch-fallback`。
- bundle SHA256：`ea2256db54f865d9be17141c158b72fead382fa14d549d919cf9389823dbf200`。
- 短 coding workload SHA256：`815488da3feff24facc9750806440fec42e17a42290f48237af74348eeaa9d95`。
- 四条 prompt 分别为 26 / 27 / 25 / 29 token，输出上限各 64；同一预分词输入、EOS 和 greedy，实际共输出 109 token。
- `max_model_len=256`、`max_num_seqs=1`、prefill 预算 64、block size=128。原生 2 个可用块加 1 个 dummy 块；Python 总块数设为 3，两侧可用容量相同。

计时排除权重校验、加载、NZ 转换、启动准备、图捕获与显式 warmup，包含调度、prefill、生成和同步读回。首次遇到新 prompt 长度时，原生执行计划的一阶段准备发生在请求内；不得把这部分从首次测量中扣除。热态结果须另外说明，不能把中间整层 padding 版本的 55 token/s 当作最终逐算子对齐版本的结果。

## 已保留的失败与修复

1. 旧逐行验证版本为 109 token / 118.13 秒，即 0.923 token/s。原始 Graph 报告保持不变。
2. 首版批量 NZ 路径虽输出正确，但草稿接受率明显下降。共享 K/K 的原地 RoPE 是实际错误，改为独立输出，并增加非零旋转、源输入不变及动态 Graph 测试。
3. 一个连续单行切片把底层存储剩余容量当作 ATB `dataSize`，导致 ReshapeAndCache 拒绝执行。ND 改为逻辑有效字节数，NZ 保留物理存储字节数，并覆盖偏移切片到 PagedAttention 的回归。
4. 旧边界测试在 3 行查询处失败：NZ 掩码没有补齐行维。现在物理 mask 行列分别按 16 对齐，真实 qlen 保持不变。
5. 整层补齐版本曾测得 eager 54.36 / Graph 55.00 token/s，但它与 Python 的 prefill 矩阵形状不同。该中间性能记录保留，不能覆盖后续结果。
6. 改为真实行数后，首次长度相关的执行计划准备使短请求首遍降至约 33 token/s。此时四条最终输出、13 轮接受长度均与 Python 相同，但 coding-1 首轮被拒绝的草稿尾部仍有差异；完整 trace 不能标为通过。
7. 独立 151936 词表平票探针中，ACLNN ArgMax 对 FP16 / FP32 均返回较小索引 1378，不能把剩余差异归因于未经验证的 argmax 平票问题。
8. logits 诊断得到明确见证：coding-1 首轮草稿第 14 个位置，Python 的 1378 / 46186 均为 20.328125，原生为 20.328125 / 20.34375。第 15 行 verify 差异发生在已经不同的草稿输入之后，不能把两个 verify 分数直接当成相同输入的目标模型误差。NPU 常量生成修复单独保存后续复测结果；不以“平票”自动豁免严格差分。

原始日志和逐轮 JSON 位于开发机 `results/ascend-python-parity-20261019/` 对应证据归档。板端目录为 `/data2/FlashEngine.cpp-deploy-b87b818-20261019/parity-20261019/`；板端系统时钟为 2000 年，文件名日期与实验记录日期以开发机为准。

## 构建与复现

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/9.0.0/atb/set_env.sh
export TE_PARALLEL_COMPILER=1 CANN_KNOWLEDGE_BANK_PROCESS_NUM=1
cd /data2/FlashEngine.cpp-deploy-b87b818-20261019
cmake --preset ascend310p -DFE_DEVICE_TESTS=ON \
  -DFE_ATB_ROOT=/usr/local/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1
cmake --build build/ascend310p -j4
ctest --test-dir build/ascend310p --output-on-failure
build/ascend310p/flashengine-native benchmark bundles/qwen3-8b-dflash \
  --workload benchmark-graph-20261019/workload.json \
  --batch-size 1 --warmup 1 --repeats 2 --trace-rounds --graph
```

第二遍的分请求墙钟与首遍分别统计；总 `output_tps` 仍包含两遍，不能替换为第二遍数值。板卡未来时间戳导致 Make clock-skew，实际构建期间明确触碰变更源文件并核对编译、链接日志；不能只看 build 命令退出码判断旧二进制已经更新。

测试服务示例：

```bash
build/ascend310p/flashengine-native serve bundles/qwen3-8b-dflash \
  --host 0.0.0.0 --port 8000 --served-model-name Qwen3-8B-DFlash \
  --max-model-len 384 --max-num-seqs 4 \
  --max-num-batched-tokens 96 --max-num-partial-prefills 3 \
  --kv-block-size 128 --kv-blocks 12 --reserve-output 64 --no-think --graph
```

Python 对照必须保留 `set_env.sh` 设置的 `PYTHONPATH`。此前清空该路径使 CANN Python 组件缺失，产生编译选项报错；此错误属于对照脚本环境，不能当作 Python 模型失败。
