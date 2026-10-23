# 310P1 Qwen3-8B + DFlash Graph 验证与单轮基准

2026-10-19，复用 [板端部署](ASCEND310P_BOARD_20261019.md) 的 310P1 / AArch64 / `vllm018` 环境和二进制。仅扩展设备测试到 Graph，未改变模型运行实现。

## Graph 验证

- CTest 三项通过；动态 KV 设备测试新增 Graph 捕获和两次回放，在单行、多行输入下改变槽位索引及数值，并核对未选中行保持。
- 完整 Qwen3-8B + DFlash 使用与 eager 相同的代码请求、FP16、256 上下文、单请求、32 输出 token、k=15、2 个 128-token KV 页。
- `--graph` 运行退出 0，32 个输出 token、prompt IDs、结束原因和全部 draft/verify/接受/提交/上下文轮次与 eager 严格一致。
- 实际草稿图回放 5 次、验证图回放 80 次。普通 decode 回放为 0，因为此次投机 decode 全走 verify 图；eager 计数 26 对应逐 token prefill。
- 本实现仍是目标单行顺序验证图与草稿块图，不是融合批量 verify，也不是整轮单图。

原始结果：`results/ascend310p-graph-benchmark-20261019/graph-validation.json`、`generate-graph.json` 和 `ctest-graph-kv.log`。

## 单轮 benchmark

4 条短代码请求：加法、反转字符串、偶数判断、两数取最大值。prompt 长度依次为 26、27、25、29 token，已冻结 token IDs，第一条与 eager/Graph 验证的 token IDs 完全一致。启用 no-think，每条最多 64 输出 token，遵循 EOS；batch=1，预热第一条一次，测量一轮。不是官方完整数据集或 HTTP 连续并发基准。

| 指标 | 结果 |
|---|---:|
| 实际测量请求 | 4 |
| 输出 token | 109 |
| 测量墙钟 | 118.1325 s |
| 聚合输出吞吐 | 0.9227 token/s |
| TTFT p50 | 9992.72 ms |
| TTFT p95 | 10833.78 ms |
| 测量期草稿图回放增量 | 13 |
| 测量期验证图回放增量 | 208 |
| 抢占 | 0 |

计时排除 bundle 校验、权重加载、算子/图准备和预热；整个进程墙钟为 257.21 s。测量期非零回放通过 `runtime_after - runtime_before` 验证，未把预热回放计入。输出 token 是实际提交序列的长度，包含停止用 EOS，不是草稿提案数。

| 请求 | 输出 token | TTFT (s) | E2E (s) | 结束原因 |
|---|---:|---:|---:|---|
| 加法 | 64 | 9.8016 | 51.7124 | length |
| 反转字符串 | 12 | 10.1839 | 22.1103 | stop |
| 偶数判断 | 15 | 9.4398 | 15.4516 | stop |
| 两数取最大值 | 18 | 10.9485 | 28.8577 | stop |

测量共 13 轮，计算 195 个提案；原始验证接受 96 个，其中实际输出接受 token 为 94、bonus 为 11，另有 4 个 prefill 首 token。按 EOS/max_tokens 截断后，每轮实际 decode 输出为 `(94+11)/13 = 8.0769` token。不能把末轮未输出的已验证后缀加入输出吞吐。

结果文本包含正常的四个函数；加法请求在 64 token 上限截断。结束后 running/waiting/active 和 used_blocks 均为 0，Prefix 没有命中（所有 prompt 短于一整页），模型进程已退出，NPU 利用率回到 0%。

## 复现命令

```bash
docker exec -it vllm018 bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TE_PARALLEL_COMPILER=1 CANN_KNOWLEDGE_BANK_PROCESS_NUM=1
cd /data2/FlashEngine.cpp-deploy-b87b818-20261019
build/ascend310p/flashengine-native benchmark bundles/qwen3-8b-dflash \
  --workload benchmark-graph-20261019/workload.json \
  --batch-size 1 --warmup 1 --repeats 1 --graph --trace-rounds
```

workload 固定 FP16、TP=1、上下文 256、最大序列 1、prefill 预算 64、partial-prefill 1、KV 页大小 128、页数 2、Prefix/chunked-prefill 开启。k=15 来自 bundle。

## 证据与限制

本地完整结果位于 `results/ascend310p-graph-benchmark-20261019/`，板端位于部署目录的 `benchmark-graph-20261019/`，包括 workload、完整逐轮 trace、摘要、解码文本和命令。

- 二进制 SHA256：`0eefa20405d508d61ddc63643556cead70cadd7b84cfb42de2b05a4bdbaa9d0e`。
- Bundle manifest SHA256：`ea2256db54f865d9be17141c158b72fead382fa14d549d919cf9389823dbf200`。
- Workload SHA256：`815488da3feff24facc9750806440fec42e17a42290f48237af74348eeaa9d95`。

这只是 ND 参考实现的单请求 Graph 正确性与短负载测量。没有运行非投机对照，不能给出投机加速比；没有 Python oracle 差分、长上下文、多并发、Prefix 命中或其他模型的 Graph 验收。保持 candidate / production_qualified=false。

板端时钟仍显示 2000 年，因此报告的系统 Unix 时间不能解释为真实日历时间；墙钟和延迟使用 steady_clock，不受该日历偏差影响。
