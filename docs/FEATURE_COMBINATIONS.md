# 0.4 功能组合与默认 Prefix

以下是原生实现范围，不是 production 资格。0.4 历史 CUDA 报告见 `VALIDATION.md`。本轮为 CoreX/310P 源码对齐增量：310P 新增 Qwen3.5/EAGLE3/Quark 参考路径，共享 scheduler 的 chunked prefill 配置和统计也已补齐；按要求没有构建或测试，新源码不受旧报告覆盖。完整范围见 [源码对齐说明](BACKEND_PARITY_SOURCE.md)。

| 目标 | 普通 | DFlash1 | EAGLE3 | Prefix | Graph |
|---|---|---|---|---|---|
| Qwen3 dense | 有 | 有 | 有 | 默认开，可与草稿组合 | decode、draft、verify、EAGLE refill 分图 |
| Qwen3 MoE | 有 | 有 | 有 | 默认开，可与草稿组合 | 同上；专家固定形状参考路由 |
| Qwen3.5 dense 文本 | 有 | 有 | 不支持 | 默认开，含 GDN 边界快照 | 单行 decode/verify 顺序回放，draft 固定块图 |
| Qwen3.5 MoE 文本 | 有 | 有 | 不支持 | 默认开，含 GDN 边界快照 | 同上；不是多行融合 target verify |

CoreX 共用上述模型代码；310P 使用独立 ACLNN 参考实现，所有 target decode/verify 都按单行顺序 Graph 执行，DFlash 草稿为固定块图。CoreX/310P 均未编译、未真机验收；表中的“有”只表示源码路径。

所有组合仍为 TP=1、greedy；需要同宽、同词表及合法 aux 层的适配草稿。不能把一个 Qwen3 检查点的训练草稿直接当作任意 Qwen3.5 检查点的适配草稿。契约 0.4.0 / bundle 1.3.0，兼容读取旧版本。

目标模型和草稿方法由启动时的 bundle 选择，不是同一引擎内按 HTTP 请求动态切换模型或草稿。

## Prefix 默认行为

CLI、HTTP 和 benchmark 都默认 `enable_prefix_cache=true`，不因启用草稿而关闭。`--no-prefix-cache` 或 workload 显式 false 只用于诊断/A-B。启动 resolved config 报告实际值。

只复用完整页，至少保留一个输入 token 重算 logits。目标 KV 与附属状态必须同时有效；缺状态就安全 miss，不得误用目标页或静默关闭总开关。短于一页的 prompt 不命中是正常行为。

本轮未验证源码进一步将附属快照绑定到实际物理末页；页面重算时刷新，回收时按该页驱逐，避免同 token 前缀的不同物理副本误用状态。目标 Prefix 仍按 token 匹配。细节见 `BACKEND_PARITY_SOURCE.md`。

`recurrent_prefix_cache_bytes` 默认 256 MiB，CLI 对应 `--recurrent-prefix-cache-bytes N`。它限制模型实例内的 CPU 附属快照；不含活动请求状态、临时复制和 EAGLE 当前半页 aux 缓冲。预算不足、被 LRU 驱逐或物理页回收时可以 miss；0 预算不伪造命中。host snapshot 报告 saved/restored/evicted/bytes，读取不会同步设备。

Qwen3+DFlash 的草稿上下文投影 KV 跟随目标物理页，共享的完整上下文页不被提案覆写。EAGLE3 的 shifted 行还依赖下一 token，故为每个活动请求单独分配草稿 KV。命中前缀时从有界 CPU aux 快照重建 shifted 草稿行，不共享可变草稿页。目标仍跳过已缓存的 prefill；草稿重建成本会单独记录。

## Qwen3.5+DFlash 的状态提交

目标先捕获所需层输出 aux。DFlash 在图外追加已提交上下文投影，然后生成固定块提案。target 按因果顺序验证 anchor 与提案，每行保存卷积历史和 FP32 SSM；最长前缀接受 m 个提案后恢复验证行 m 的状态。bonus 尚未消费，留给下一轮；未接受的 recurrent 状态不得留在主槽。

full-attention KV 的未接受尾部是无效 scratch，下一轮会覆盖，不能发布到前缀缓存。只有已计算、已提交的完整前缀可发布附属 GDN 快照。每批先恢复所有已获准命中的请求，再允许新快照驱逐；完成、取消、抢占释放活动槽，恢复时使用快照或重算历史。

Qwen3.5 的 draft 使用 ATen 固定上下文参考 attention，支持合法不同 head_dim；target decode/verify 复用固定单行 Graph，按请求和验证行顺序回放。状态复制、接受逻辑和元数据更新在图外，不冒充整轮单图或高性能多行 verify。

## 打包与运行

```bash
python3 -m scripts.export_bundle /models/Qwen3.5-MoE /bundles/qwen35-moe-dflash \
  --draft /models/compatible-DFlash --spec-num-tokens 2 \
  --verifier ../FlashEngine.cpp/build/cuda/flashengine-native
../FlashEngine.cpp/build/cuda/flashengine-native serve /bundles/qwen35-moe-dflash \
  --max-model-len 2048 --max-num-seqs 4 --kv-blocks 32 --graph --no-think
```

无需另加 `--prefix-cache`。容量必须能容纳目标/草稿权重、KV、Graph workspace、GDN 主槽与逐行回滚暂存。所有 MoE 专家仍驻留设备；没有专家卸载。

## 回归入口

在 Python 仓运行，输出路径必须尚不存在：

```bash
python3 -m scripts.check_native_combinations ../FlashEngine.cpp/build/cuda/flashengine-native \
  /fixtures/combinations /reports/combinations
python3 -m scripts.check_native_combinations ../FlashEngine.cpp/build/cuda/flashengine-native \
  /fixtures/combinations /reports/combinations-pressure --pressure
```

覆盖四种目标的普通路径及六种草稿组合；ON 测试故意不传开关，核验真正默认值。检查实际命中/附属状态恢复、Graph 计数、ON/OFF 与 eager/Graph 输出、活动状态归零。宽松容量比较逐轮 trace；抢占压力比较最终输出，因为重算会改变轮次切分。合成模型不证明训练准确性、接受率或加速收益。

## 仍未关闭的门禁

真实 EAGLE3 的 Python/C++ 逐轮差异、旧版 BF16/长输出数值差异仍需收敛。Qwen3.5+DFlash 真实训练草稿、长上下文、长稳、性能门禁需分别完成；随机合成草稿不能替代。CoreX/310P 没有 SDK/真机报告；不因契约或共用源码存在就宣布这些后端已通过。
