# 原生 CUDA 基准与严格验收

本程序是 C++20 部署实现，不在部署进程内加载 Python。Python 只在开发验收机生成冻结 token workload、oracle 和对比报告。基准结果不自动授予 production。

## 单独运行原生基准

```bash
build/cuda/flashengine-native benchmark /bundles/qwen3 \
  --workload /reports/workload.json --batch-size 1 --warmup 3 --repeats 1 --graph
```

JSON 输出位于 stdout；日志位于 stderr。`--eager` 关闭图，`--trace-rounds` 导出 DFlash 逐轮 trace。workload 的 dtype、上下文、最大序列、prefill 预算、partial-prefill、KV 页大小/页数、每请求 EOS/输出长度是权威输入；bundle SHA256 必须匹配，不静默裁剪。输出包含 resolved config、请求 token IDs/TTFT/E2E/完成原因、测量前后 Graph 计数、总墙钟和吞吐。

计时不包含模型加载、分词和预热。batch=1 是串行；batch>1 是一组结束后再发下一组，不等于持续 HTTP 并发。最终 token 和每轮 draft/verify/接受/提交/上下文都必须检查；吞吐快不能豁免数值差异。

## Python / C++ 公平对比

在对应 Python oracle 仓运行 `scripts.make_native_workload` 冻结输入，`scripts.run_native_benchmark` 顺序调度 ABBA 两轮，`scripts.audit_native_benchmark` 检查配置、SHA256 与测量期 Graph。HTTP 使用 respect-eos 的独立 workload 及 `scripts.run_native_http_benchmark` / `scripts.audit_native_http_benchmark`；SSE chunk 数不是 token 数。

完整命令和当前实测表位于 Python 仓 `docs/BENCHMARK_NATIVE.md`。本地两仓布局是父仓 `FlashEngine/` 和独立 `FlashEngine/flashengine-cpp/`；分开发版后需从与契约锁匹配的 oracle 版本取得离线工具。

## 当前严格边界

2026-10-17：Qwen3-0.6B FP16、8 条×128 token 的 Python/C++ 串行测试仅 7/8 严格相同。teacher-forcing 在相同前缀定位到第 22 个 token 的一个 FP16 ULP 差异：Python margin 非零、原生 top-2 平票。eager/Graph 均复现；没有全局宽松开关，仍是失败。

显式把 Python attention 也切为 FlashInfer 后，同一 8 条×128 token、batch=1 的 eager/Graph 对照均 8/8 严格相同；top-8 logits 仍非位等。此诊断不替换默认 FlashAttention 的主基准、不覆盖失败、不扩展到其他配置，全部后端仍 candidate。

合成 MoE+DFlash 的 batch 1/2/4/8、eager/Graph、chunked prefill、KV 抢占、上下文尾部和 EOS 夹具通过严格 token/trace 对比。合成高 margin/受控提案只证明已测状态行为，不证明训练 MoE 模型质量、真实接受率或投机加速。

所有结果必须带各自二进制和源码哈希；改动后不要替换旧报告内的哈希。CUDA 外的 AArch64、CoreX 和 310P 尚无本次设备报告；所有后端仍为 candidate。细节见 [验收记录](VALIDATION.md) 与 [部署手册](DEPLOYMENT.md)。
