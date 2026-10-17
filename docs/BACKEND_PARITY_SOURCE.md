# 三后端源码对齐与 Python 功能同步（2026-10-17）

本轮遵照要求仅修改源码：**没有构建、运行测试、生成新二进制或取得真机验收**。旧 `build/` 内的程序尚未包含本轮变更，`VALIDATION.md` 中既有 CUDA 报告只对应历史二进制。所有新增路径均为 source-prepared，不能写成验证通过或 production。

本轮也调整了 CUDA/CoreX 共用路径的附属快照绑定；不是只改 310P 文件，因此此前 CUDA 通过报告同样不能覆盖新源码。

## 1. 对齐的是行为和状态，不是厂商算子性能

| 源码能力 | CUDA | CoreX | Ascend 310P |
|---|---|---|---|
| Qwen3 dense/MoE 普通与 DFlash | 既有实现 | 共用 CUDA 模型层，ATen 参考算子 | ACLNN 统一参考执行 |
| Qwen3 dense/MoE + EAGLE3，含 d2t | 既有实现 | 共用相同状态机与私有 KV 管理 | 新增 shifted 草稿流、verify/refill/carry |
| Qwen3.5 dense/MoE 普通与 DFlash | 既有参考实现 | 共用 hybrid/block draft 实现 | 新增 GDN、共享专家、逐行 verify 和回滚 |
| Prefix 默认开启，可组合投机 | 既有状态门控 | 共用 CPU 快照与重建 | 新增有界 CPU aux/GDN 快照和就绪门控 |
| decode/draft/verify Graph | 既有分图 | 厂商兼容 C++ Graph | 固定地址 ACL Graph；target 按单行顺序执行 |
| Quark int8 检查点 | 加载期反量化 | 加载期反量化 | 新增逐通道 `(q-zp)*scale` 装载为 FP16 |
| 公共 scheduler / HTTP/SSE / 取消 | 共享 | 共享 | 共享 |
| 本轮验证状态 | 未重建、未重跑 | 无 SDK，未编译、未上板 | 无 SDK，未编译、未上板 |

310P 仍为 FP16 计算、FP32 GDN SSM。其他两后端保留 FP16/BF16。三后端仍要求 TP=1、greedy；Qwen3.5+EAGLE3 仍拒绝。Quark 支持范围以已冻结 bundle/profile 校验为准，**加载期反量化不是真 int8 GEMM**。

CoreX 不需要复制一套模型：`cuda_model.cpp`、`hybrid_model.cpp`、`block_draft.cpp` 是同一份源码；`portable_gpu_ops.cpp` 替代 NVIDIA kernels/FlashInfer。不能把厂商 ATen 参考路径称为 ixformer 原生高性能 attention。310P 的 KV 仍是本实现 ND 布局，ATB/NZ 优化未实现。

## 2. 310P 实际补充的状态路径

`ascend_net.hpp` 构造纯 ACLNN 执行程序，`ascend_model.cpp` 管理请求状态和图外提交。没有 TorchNPU、Python/Rust 或 Python JIT 运行时。

- Qwen3.5：partial RoPE、门控 full attention、GDN 卷积历史与 FP32 SSM、dense/MoE 与 shared expert；每个 request ID 占独立状态槽。每个验证行备份状态，接受 m 个提案后恢复行 m，bonus 保持未消费。
- DFlash：固定块非因果草稿、aux 投影、目标逐行验证、最长前缀接受、只投影已接受上下文；KV 物理页与目标页对应，提案不得覆写已发布前缀。
- EAGLE3：每个活动请求单独划分草稿 KV；shifted 行使用 `token[p+1]` 与 `fc(aux[p])`。verify 后 refill 行 m 提供下一轮 carry，处理 d2t 缩减词表映射。完整 Prefix 页保存 target aux，命中后按当前 prompt 重建 shifted 草稿行。
- Prefix：CPU 快照受 `recurrent_prefix_cache_bytes` 默认 256 MiB 预算控制；缺少状态安全 miss，保持默认开关不变。目标页回收时删除关联快照；每批先恢复已准入请求，再生成可能驱逐旧条目的新快照。
- 附属快照按累计 token 前缀及实际物理末页绑定，`prefix_page_ready` 在 probe/publish 时校验对应页。页面重算时刷新对应快照；不同物理页上的同一 token 前缀不会互相覆盖附属状态。无 GDN 层的全注意力 Qwen3.5 不依赖空的 recurrent 快照，避免产生无字节预算约束的空条目。目标 Prefix 的语义匹配仍只使用 token IDs，不把物理页 ID 暴露给外部契约。
- 生命周期：结束、取消和抢占统一释放请求槽、carry 和半页 aux；私有草稿 KV 在重建/重新预填充后才允许读取。读取 host snapshot 不调用 ACL 同步。
- Graph：planner/executor/workspace 创建在捕获外；输入地址固定，元数据每次重写，dummy 页与 GDN dummy 槽独立。target 每次只有一行；草稿块固定 k+1 行。CPU 快照、接受和回滚在图外。它不是 CUDA 的融合批量 verify，也不是整轮单图。

ACLNN 基础层同时补充了非连续 tensor 的真实 materialize、broadcast 形状校验、2D/3D GEMM 分发和 GDN 所需向量运算。310P 的 FP32 cube 限制不能绕过，因此 SSM 更新使用 FP32 向量乘/归约，不把 FP32 SSM 当作 FP16 GEMM 状态。

接口依据为官方 [BatchMatMul](https://www.hiascend.com/document/detail/en/canncommercial/850/API/aolapi/context/ops-nn/aclnnBatchMatMul.md) 和 [Softplus](https://www.hiascend.com/document/detail/en/canncommercial/850/API/aolapi/context/ops-nn/aclnnSoftplus.md) 等公开 ACLNN 文档。接口存在不等于目标 310P/SDK 上所有算子可用；未来仍须核对所选 CANN、驱动、产品支持及 Graph 可捕获性。

## 3. 同步 Python 的公共调度能力

chunked prefill 原先已由所有后端共用 `Scheduler` 执行。本轮补齐外部配置和统计，不为三个后端复制三套 scheduler。

| Python 语义 | C++ 行为 / 配置 |
|---|---|
| `enable_chunked_prefill=True` | 默认开启；新增 `--enable-chunked-prefill` / `--disable-chunked-prefill` |
| `max_num_batched_tokens` | 每步 prefill token 预算；超长 prompt 分多步处理，未到末块不采样输出 |
| `max_num_partial_prefills` | 保留同名限制；partial prefill 持有 KV/投机状态并计入 active/max_num_seqs |
| 公平轮转、同质 prefill/decode 批次 | 保留共享实现；有 waiting 时交替选择 prefill/decode，没有进展立即尝试另一侧 |
| 连续批处理、KV 抢占重算 | 请求可动态加入；抢占释放模型附属状态，再从 Prefix 或完整历史恢复 |
| 取消、EOS/max_tokens/上下文硬停 | 公共提交/释放路径；投机轮在剩余上下文不足时回到单 token |
| scheduler 统计 | 新增 steps、prefill/decode steps/tokens、chunked steps、admitted、partial_prefills |

`benchmark` 现在真正读取 workload 中的 `enable_chunked_prefill`，不再忽略 false 而使用默认 true。关闭分块时，token 预算必须不小于 `max_model_len`，否则拒绝启动。Prefix 探测后若请求未获准入，会释放临时共享页引用，避免等待中的新请求占住 KV。

未来在编译并验收新部署物后，三后端使用同一组参数（示例**本轮未执行**）：

```bash
/path/to/flashengine-native serve /bundles/compatible-model-and-draft \
  --max-model-len 2048 --max-num-seqs 4 \
  --enable-chunked-prefill --max-num-batched-tokens 512 \
  --max-num-partial-prefills 2 --kv-blocks 32 --graph --dtype float16
# 禁用分块的对照；max_model_len 仍为 2048
/path/to/flashengine-native serve /bundles/model \
  --max-model-len 2048 --disable-chunked-prefill \
  --max-num-batched-tokens 2048 --kv-blocks 32 --dtype float16
```

Prefix 无需额外开关；`--no-prefix-cache` 仍只给显式 A/B。KV 页数表示可供请求使用的页，后端额外分配 dummy 页，不可把 Python 310P 的原始含 dummy 池数量直接当作原生 usable 页数。

## 4. 同步的观测项

`/metrics` 和 `/v1/stats` 继续读取同一个主机快照，新增：

- `flashengine_scheduler_{steps,prefill_steps,decode_steps,prefill_tokens,decode_tokens,chunked_prefill_steps,admitted_requests}_total`
- `flashengine_partial_prefill_requests`
- `flashengine_prefix_state_{saved,restored,evicted}_total`、`flashengine_prefix_state_bytes/entries`
- `flashengine_spec_draft_tokens_total`、`flashengine_spec_accepted_tokens_total`
- `flashengine_spec_position_N_proposed_total`、`flashengine_spec_position_N_accepted_total`

投机统计按模型实际验证轮计数；末轮遇到 EOS/max_tokens 截断时，轮内接受数不等于最终发给客户端的 token 数。`generated_tokens_total` 仍只统计实际提交输出，不能把接受计数冒充客户端吞吐。

`capabilities` 和 stats 的 implementation revision 标识为 `backend-parity-source-2026-10-17`，status 为 `unvalidated-source-changes`。它说明源码路径，不授予发布资格。契约 0.4.0 和 bundle 1.3.0 未改动，也未人工修改 vendor 快照。

## 5. 仍未同步的 Python 能力与后续门禁

没有实现 DSpark、DFlash2、随机采样/top-k/top-p、TP>1、Python WebUI、通用 tokenizer、真 W8A8/GPTQ/AWQ/FP8 或 MoE 专家卸载。没有实现 CoreX ixformer 原生 attention、310P ATB/NZ 高性能路径。新增源码不能覆盖旧真实 EAGLE3、BF16 和长输出的严格失败，也不能宣称性能达标。

后续需分别构建并验收三个后端：普通 eager → 普通 Graph → spec eager → spec Graph，再覆盖 Prefix+chunked prefill、混合生命周期、受控接受长度、上下文尾部、取消/抢占、内存、长稳和热态 A/B。此次**没有执行上述任何步骤**，也没有更新已有 build 或测试工件。
