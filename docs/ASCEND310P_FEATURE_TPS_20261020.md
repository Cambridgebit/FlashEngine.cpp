# Ascend 310P：其他模型功能与 TPS 实测（2026-10-20）

本报告记录 `192.168.16.118` 的 Ascend310P1、容器 `vllm018` 上的实际运行。**测试已按用户要求暂停，队列未完成；本轮模型进程与控制器已停止，日志保留。** EAGLE3 与 Qwen3.5 MoE 按用户要求跳过。测试没有更改推理源码，也没有将失败检查点改写成兼容配置。当前 MoE / Qwen3.5 使用 ACLNN ND 参考实现，不能沿用 Qwen3 dense ATB/NZ 路径的性能验收结论。

二进制 SHA256：`aabf6e2aa8d3184ddf75c21c7a141a44bcce8150b7cbaa8fd3016958d72259a5`。部署根目录：`/data2/FlashEngine.cpp-deploy-b87b818-20261019`。板端时钟不准，报告日期使用开发机时间。以下结果仍是 candidate 的定点测试，不是发布或 production 验收。

## 第一组：32 token 编码负载

两个英文编码 prompt，每个重复两次，每次固定生成 32 token，ignore-EOS，总输出 128 token；batch=1、warmup=1、FP16、TP=1、greedy、thinking off。上下文 256、max_num_seqs=2、prefill 预算 64。TPS 为总输出 token / 主跑外层墙钟，包含 prefill，不包含加载和 warmup；没有使用 decode-only 指标替换整体吞吐。

| 目标模型 | C++ eager TPS | C++ Graph TPS | Python eager TPS | Python Graph TPS | 输出校验与限制 |
|---|---:|---:|---:|---:|---|
| Qwen3-0.6B | 62.133 | 81.123 | 14.677 | 79.902 | 两种模式各 4/4 请求最终 token、结束原因一致 |
| Qwen3-Coder-30B-A3B-Instruct | 超时 | 待短负载定位 | 5.809 | 未完成 | C++ 原始 eager 限时 600s 后强制退出，无有效 TPS |
| Qwen3.5-4B | 3.006 | 3.125 | 首次配置识别失败 | 补配置适配后设备故障 | C++ eager/Graph 仅 2/4 一致，另两条从第 29 个输出 token 分歧 |
| Qwen3.5-9B | 1.501 | 1.513 | 3.332（无效配置基线） | 跳过 | C++ eager/Graph 4/4 一致；Python 配置适配错误，需复测 |

0.6B 主跑 decode Graph replay 增量为 124，证明使用了真实回放；Graph TPS 约为 Python 的 101.5%。这项最终输出比较不等于完整中间 trace 验收。

4B 主跑 Graph replay 同样增加 124，但回放不代表数值一致。4B 两条失败均为素数判断 prompt 的两个重复，从索引 28 分歧；没有以 FP16 平票为理由豁免。9B 的旧 Python 输出包含异常重复，4/4 请求从首 token 与 C++ 分歧。已查明测试配置适配存在错误：旧 Transformers 不识别 `qwen3_5`，控制器注册的本地配置类将嵌套 text_config 转成普通 PretrainedConfig；9B 的 text_config 未显式包含 tie_word_embeddings，而顶层为 false，适配未继承顶层值，普通 PretrainedConfig 默认 true，导致错误绑定输出权重。其 3.332 TPS 与输出不能作有效 oracle。适配 v2 已补齐该字段继承，尚未复测就按用户要求暂停。4B 的嵌套配置显式为 true，不受这一漏传影响；其 Graph 设备故障与 C++ 模式分歧仍保留。

Qwen3.5 的初始 workload 指定 KV block_size=128，Python 因 head_dim=256 自动调整成 64；两侧可用页数均为 8，因此实际池容量分别为 1024 与 512 token。这一组不能称为完全相同的最终配置。后续短负载显式指定 64，独立保存结果，不覆盖原记录。

Python 4B Graph 退出 134，日志报告 `paged_attention_decoder_mask_0` 的 MTE DDR 地址越界与运行时错误 507011。9B Python Graph 因相同 head_dim 的先前故障而主动跳过；不能将其记为一次已执行的失败。设备后续恢复 Health=OK。

## 真实 DFlash 检查点兼容性

`/data/Qwen3.5-4B-Dflash` 与 `/data/Qwen3.5-9B-DFlash` 的导出均被拒绝：草稿包含滑窗 attention，当前原生 profile 不支持该形态。保留原始错误 `当前 profile 不支持滑窗注意力`，没有删掉 sliding_window/layer_types 绕过检查。故这两个真实检查点没有有效 C++ TPS，不能用源码宣称的 Qwen3.5+DFlash 组合代替实测。

## 优先队列：短负载定位

MoE 普通与 DFlash 优先，其后为 Qwen3.5 dense。短负载使用单条 `Write a Python factorial function.`，固定 8 token，warmup=1、repeats=2、batch=1，总输出 16 token；上下文 128、max_num_seqs=1、prefill 预算 32、2 个可用 KV 页。Qwen3.5 页大小显式为 64，其余为 128。该组用于定位加载/执行/兼容性问题，不能替换上面的 32 token 结果，也不能代表稳定长生成速度。

短负载队列已暂停于 `coder30-base-short / cpp-graph`，没有产生有效 TPS，也没有执行后续项目。暂停记录保存在 `priority-tps-20261020/PAUSED.json`，当前 progress.json 的运行中标签是中断前状态，不能据此认为控制器仍在运行。原 30B 超时样本仍保留。采样显示新 30B 进程先顺序读取约 61.2GB bundle，再转换并装载 BF16 权重；约 7 分钟后才观察到明显 NPU 计算占用。原 600s 超时没有 OOM kill 日志证据，不能直接归因于显存不足。暂停后 NPU Health=OK、AICore=0%；Memory-Usage 为 NA，不报告显存归零。

## 慢路径的源码原因

- `src/ascend_model.cpp` 的参考 prefill 逐 token 调用 target，无法获得批量 prefill 的效率。
- `src/ascend_net.hpp` 的 MoE 每层依次处理 top-k 专家，执行权重 gather、matmul 和累加，没有使用融合 MoE 算子。
- 参考 DFlash 的 target verify 仍逐行执行，辅助隐藏状态有设备同步与主机读回；不是已优化 8B 的批量 verify 和整轮 Graph。
- Qwen3.5 使用参考 GDN/attention，Graph 是单行回放。实测 4B/9B 的 Graph 吞吐提升很小，且上述正确性问题尚未关闭。

## 证据

原始 32 token 结果位于开发机 `results/feature-tps-20261020/stage/feature-tps-20261020/`；归档 `feature-stage.tgz` 包含成功与失败日志。板端对应 `feature-tps-20261020/`。短负载单独保存在板端 `priority-tps-20261020/`，包含逐请求 token、配置、进程退出码、运行命令、资源采样与进度记录。目录中的 `analysis.json` 是从原始逐请求 JSON 计算的汇总，不改写原始测量。
