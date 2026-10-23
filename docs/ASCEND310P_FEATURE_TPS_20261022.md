# 310P C++ 补充测试与暂停记录（2026-10-22）

本轮已按用户要求暂停。停止检查返回本轮模型进程 `remaining=[]`、控制器 `controllers stopped=[]`，不再补跑。板卡仍有其他负载，最后一次 NPU 读数 AICore 为 97%；不能将本轮停止解释为全板空闲。

## 本轮实测结果

推理入口为纯 C++ `flashengine-native`；Python 用于导出、测试控制及独立行为对照。下表 TPS 为预热后的主测试输出 token / 墙钟，包含 prefill，不包含权重加载和预热。均为 FP16、TP=1、greedy、关闭 thinking、串行请求，每请求固定生成 32 token。短样本结果不能扩大为完整模型验收。

| 模型 | C++ eager TPS | C++ Graph TPS | Python TPS | 输出校验 |
|---|---:|---:|---:|---|
| Qwen3-Coder-30B-A3B + DFlash | 未测 | 1.648 | 34.399（spec Graph） | 2/2 请求最终 token 及结束原因一致 |
| Qwen3.5-4B | 3.047 | 3.068 | 3.960（eager） | C++ 两模式 4/4 一致；对 Python 仅 2/4 一致 |
| Qwen3.5-9B | 1.713 | 1.699 | 未完成 | C++ 两模式 4/4 一致；缺少有效 Python 对照 |

Coder 使用一条 coding prompt、重复两次，共 64 输出 token；Qwen3.5 使用两条 coding prompt、重复两次，共 128 输出 token。Coder 可用 KV 块为 2×128，Qwen3.5 为 8×64；Python 分别分配 3、9 块，含一个保留 dummy 块。不同模型行不能作为投机加速比。

### MoE + DFlash

目标 `/data/Qwen3-Coder-30B-A3B-Instruct`，草稿 `/data/Qwen3-Coder-30B-A3B-DFlash`。C++ 与 Python 各完成 8 个投机轮，未裁剪提交计数均为 66，E=8.25；扣除 prefill 首 token 后，实际交付的 decode token 为 62，平均每轮 7.75。C++ 主测试草稿 Graph replay 增加 8、目标验证单行 Graph replay 增加 128。这里只完成最终输出和汇总计数对照，未采集 Python 全部逐轮草稿/验证轨迹。

C++ 总 TPS 仅为 Python 的约 4.8%。C++ 使用 ACLNN ND 参考 attention、逐专家 gather/矩阵乘/累加，目标验证按单行顺序执行，含主机侧辅助隐层处理；Python 使用 NZ 分组 GEMM 和整轮投机 Graph。Graph 有实际重放，但尚未达到 Python 的执行结构与性能。

### Qwen3.5 dense

4B 的 prime prompt 在两个重复中均从第 29 个输出 token 开始与 Python 分歧；eager 和 Graph 都出现，尚未通过 logits 定位原因，不能直接认定为浮点近似。当前 block_size=64 下 C++ 两模式一致，不覆盖历史 block_size=128 的失败记录。

4B C++ Graph 首 token p50 为 4544ms，Python eager 为 759ms；纯 decode 分别为 5.267 和 4.234 TPS。整体吞吐的差距主要体现在首 token 阶段。

9B C++ Graph 首 token p50 为 8184ms、纯 decode 为 2.912 TPS。4B/9B 主测试 decode Graph replay 均增加 124，但整体性能未有明显提升。9B Python 对照在启动前被其他工作负载占用检查拦截，未运行推理；历史错误绑定 lm_head 的 Python 数字无效，不能作为此处基线。无法完整排除其他负载对 9B 计时的影响，其速度仅作为当前观察值。

## 未完成或拒绝项

- Qwen3.5-4B/9B 的现有 DFlash 检查点被 sliding attention 配置校验拒绝，未取得有效 C++ TPS；未改动检查点绕过校验。
- 非 Coder `/data/Qwen3-30B-A3B-Dflash` 配置归一化失败（`hidden_size`，并含 32000 草稿词表），不能套用 Coder 的结果。
- EAGLE3 及 Qwen3.5 MoE 按用户要求跳过。
- 9B Python 对照、4B 分歧定位、MoE/3.5 性能优化均未完成；暂停期间不自动继续。

## 历史已完成范围

优化过的 Qwen3-8B dense ATB/NZ 路径已有独立报告：普通 C++ 9.483 TPS；DFlash 两遍合计 41.183 TPS，首遍 32.797、热态第二遍 55.351 TPS。DFlash 已完成 8/8 输出以及 26 轮草稿、验证、提交、上下文轨迹严格对照。上述数字来自此前测试，不能与本轮短样本直接计算加速比。详见 [8B 对齐报告](ASCEND310P_PYTHON_PARITY_20261020.md)。MoE/3.5 尚未复用同等优化路径。

## 证据与版本

- 本地结果：`results/feature-tps-20261022/evidence/`，汇总 `analysis.json`，包含七组已完成结果。
- 板端结果：`/data2/FlashEngine.cpp-deploy-b87b818-20261019/feature-tps-20261022/`，含停止记录、启动失败日志、模型和源码身份审计。
- C++ 二进制 SHA256：`1b8015328e3a8124e0f5c2ed0f2f4478ed2407853f8080033074a7a259aaf67c`。
- C++ 源码身份 SHA256：`d2847bda327c645abee42ef83f2df4e18c219e7ee00e58a8a43d01be60f64635`，52 个构建依赖文件审计未变。
- Python 源码身份 SHA256：`dd6b642dd9ba0e3745f92fff89d56828ea92cddf46d3e3531a55e069bf7738de`。
- 目标/草稿与 bundle 的 safetensors 同 inode 审计通过，目标归一化配置一致。

本轮未修改模型数值实现；复用此前加入可选启动阶段日志的 C++ 二进制。所有结论限于上述真实测试，仍为 candidate。
