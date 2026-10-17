# 0.3 原生功能增量

本页保留 0.3 历史范围；最新组合与 Prefix 默认行为见 [0.4 功能组合](FEATURE_COMBINATIONS.md)。

部署代码仍为 candidate。契约为 0.3.0，bundle 为 1.2.0，兼容读取 0.1/0.2 契约与旧 bundle。

| 功能 | 实现范围 | 当前限制 |
|---|---|---|
| Prefix cache | Qwen3 dense/MoE 普通路径；完整页共享、引用计数、空闲页 LRU 驱逐 | 默认关闭；暂不与草稿或 Qwen3.5 递归状态组合 |
| EAGLE3 | 一个 midlayer；显式三层 aux；完整/缩减词表 d2t；固定 k=1..15 | Qwen3 dense/MoE；不承诺真实训练草稿逐轮差分已通过 |
| EAGLE3 Graph | draft 单步、target verify、draft refill 分图；request ID 保存跨轮状态 | 冷启动重算在图外；接受与元数据处理仍有 host 路径 |
| Qwen3.5 | 文本 dense/MoE、门控 attention、partial RoPE、Gemma norm、GDN、共享专家 | 参考路径；每个请求逐 token；无多模态/MTP |
| Qwen3.5 Graph | 固定单行 decode Graph；多请求依次回放；GDN 状态通过设备槽索引读写 | 不是多行批次 Graph；没有 draft verify 回滚或 Prefix 快照 |
| Quark int8 | 保留原始 int8/scale/zero-point；C++ 加载时按输出通道反量化 | 执行与常驻权重为 FP16/BF16；不减少展开后的显存，不是 int8 GEMM |

CUDA 和 CoreX 共用 ATen 侧实现。只有 CUDA 可以在本机验收；CoreX 仍缺 SDK/真机。310P 显式拒绝新 EAGLE3/Qwen3.5/Quark profile，不能把 CUDA 的实现当作 310P 已支持。

## Prefix cache 的语义

启动加入 `--prefix-cache`。匹配键是模型实例内的累计 token IDs，不靠可能碰撞的截断哈希。只发布已经计算完的完整页，活动请求引用页时不能被驱逐；共享页不再写入。即使 prompt 整页命中，也至少重算最后一个输入 token 取得 logits。

释放请求只减少引用；完整缓存页可以继续驻留。`kv_used_blocks=0` 表示没有活动引用，不表示缓存内容被抹除。`prefix_resident_blocks` 报告驻留页；`prefix_evictions_total` 报告回收次数。`/metrics` 与 stats 使用同一调度器快照。

原生 benchmark 在 warmup 和各组之间保留同一 Scheduler，可检查真实热缓存。workload 配置包含 `enable_prefix_cache`；ON/OFF 对照必须使用相同 prompt。

## EAGLE3 的检查点

目标 aux 层在 bundle 中固定，语义是 Qwen3 层输入。draft 行位置 p 消费 `tok[p+1]` 和该行条件，采用因果注意力。按 request ID 保存 carry/aux/token，完成、抢占、取消均清除；重新 prefill 后不复用旧请求状态。

草稿输出先 argmax，再执行 `draft_id + d2t[draft_id]`。bundle 装载检查 int64 映射、大小和目标范围。部署程序只读 safetensors；官方 pickle 检查点可在 Python 开发机离线转换：

```bash
cd /path/to/FlashEngine
python3 -m scripts.prepare_native_checkpoint /models/EAGLE3 /models/EAGLE3-safetensors
python3 -m scripts.export_bundle /models/Qwen3 /bundles/qwen3-eagle \
  --draft /models/EAGLE3-safetensors --spec-num-tokens 2 \
  --verifier ../FlashEngine.cpp/build/cuda/flashengine-native
```

转换使用 `torch.load(weights_only=True)`，拒绝非张量字典，保留源文件 SHA256。原生运行不加载 pickle/Python。

## Qwen3.5 的参考执行

打包器从 `text_config` 解析模型。视觉/MTP 张量在装载计划中明确标为忽略，仍校验文件结构与哈希。支持层型、GDN 投影、FP32 `A_log/dt_bias`、卷积历史、FP32 SSM、门控共享专家和文本 partial RoPE。

完整 attention 使用固定上下文形状的 ATen 参考计算；它不借用只编译 head_dim=128 的 Qwen3 FlashInfer kernel，所以可读取 Qwen3.5-2B 的 head_dim=256。每个 GDN 层只分配请求状态；只有 full-attention 层分配 KV。抢占释放请求状态槽，恢复时重算已提交历史。

此实现优先提供可检查的数学和生命周期路径，未承诺高性能。`--graph` 捕获固定单行计算，状态槽、位置和页表在图外更新；snapshot 明确报告 `graph_execution=sequential-single-row`，不冒充 batch=4/8 的融合图。

## Quark 的限制

只接受 config 中声明的 Quark 逐输出通道 8-bit int 权重。公式为 `(int8_weight - zero_point) * scale`，先 FP32，再转计算 dtype。scale 必须有限、为正并与输出通道一致；缺失 scale、孤立伴生参数和不支持的格式明确拒绝。

暂不支持 GPTQ、AWQ、FP8、分组 4-bit、msmodelslim 的 310P 真 W8A8 GEMM。不得用加载成功声称量化执行收益。

## 功能验收入口

从 Python 仓运行，所有输出使用新目录：

```bash
python3 -m scripts.create_native_profiles_fixture /models/Qwen3-tokenizer /tmp/native-fixtures
python3 -m scripts.check_native_profiles ../FlashEngine.cpp/build/cuda/flashengine-native \
  /tmp/native-fixtures /reports/profiles-check --python
```

该脚本顺序运行 CUDA/Python worker，保存原始命令和输出，比较 eager/Graph 最终 token、EAGLE3 逐轮 trace、Prefix ON/OFF、Quark 与显式反量化参考，并检查请求状态与 KV 回收。夹具不替代训练模型验收。内存检查、真实模型和 SDK 测试需要分别归档。

新版 hybrid 夹具输出随 token 在 52/53 间变化，避免恒定输出掩盖 Graph 读回旧缓冲。可用 `--profiles hybrid,hybrid-moe --python` 单独回归。真实 EAGLE3 虽最终 token 相同，逐轮 trace 仍失败；Python Graph 在该次环境回退 eager，不能据此通过图对照。具体成功/失败、二进制和报告哈希以 [验收记录](VALIDATION.md) 为准。
