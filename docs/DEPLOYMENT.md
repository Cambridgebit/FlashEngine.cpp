# x86_64 / AArch64 原生部署手册

状态：2026-10-17，candidate。部署进程为 C++，不启动 Python、Triton、Rust tokenizer 或 Python JIT。Python 只用于另一台开发机上的打包、oracle、验收与证据归档。**源码准备、SDK 编译通过、真机正确性通过、production 是四个不同状态。**

## 1. 选择部署物

| 后端 | 编译机 / 运行机 | 当前实现 | 当前证据 |
|---|---|---|---|
| CUDA | Linux x86_64 + NVIDIA；Windows 使用 WSL2 | LibTorch C++、预编译 CUDA kernels、FlashInfer 原生 paged attention、CUDA Graph | WSL + RTX 4070 实跑；部分真实模型严格差分仍失败，未获 production 资格 |
| CUDA | Linux AArch64 + NVIDIA | 同一源码，在目标架构原生构建 | 需要匹配的 AArch64 LibTorch/CUDA；尚未编译或真机验收 |
| CoreX | Linux + Iluvatar，主机架构必须与厂商 SDK/LibTorch 一致 | 厂商 LibTorch C++、固定形状 ATen paged reference、兼容 Graph | 参考算子已在 NVIDIA 回归；**未取得 CoreX SDK，未在 CoreX 编译/运行** |
| Ascend 310P | 首选 Linux AArch64 + 310P；x86_64 仅在厂商提供相应 SDK/驱动时适用 | AscendCL、公开 ACLNN、ACL Graph；不链接 TorchNPU | **未取得 SDK/板卡，源码待编译、待真机验收** |

运行 `flashengine-native capabilities` 查看二进制实际编译后端。一个 ELF 只链接一个设备运行时，不能用 `--backend` 把 CUDA 程序切成 CoreX/310P 程序。主机 preset 只能校验 bundle/tokenizer，不能进行 CPU 推理。x86_64 ELF 不能复制到 AArch64 直接运行；bundle 是架构无关的数据，可以复用。

0.4 增量见 [功能组合](FEATURE_COMBINATIONS.md)：Qwen3 dense/MoE 可组合 DFlash/EAGLE3，Qwen3.5 文本 dense/MoE 可组合 DFlash；Prefix 默认开启，可与这些草稿组合，使用状态门控/快照恢复。本轮为 310P 补充这些 profile 和 Quark 加载期反量化的 ACLNN 参考源码，并补齐公共 chunked-prefill 配置；**只改代码，没有构建/测试，原有二进制未更新**，见 [源码对齐说明](BACKEND_PARITY_SOURCE.md)。共同约束为 TP=1、greedy；DFlash1 固定 k、全词表、非因果全注意力。仍拒绝 TP>1、DSpark、DFlash2、Qwen3.5+EAGLE3、滑窗草稿和未实现的量化格式；Python 支持不代表原生已验收。

Qwen3 CUDA FlashInfer 和 CoreX 共用模型层使用 head_dim=128；Qwen3.5 与 310P 使用独立参考 attention。计算为 fp16/bf16；310P 只接受 fp16，GDN SSM 采用 fp32。MoE 支持逐专家及融合 3D 权重、mixed dense/sparse 层、top-k 路由；Qwen3.5 另有门控 shared expert。所有专家驻留设备，不提供专家卸载；30B 模型不能因只激活 3B 就按 3B 权重预算显存。

## 2. 准备离线输入

在 Python 仓所在的开发机执行，模型路径必须为本地目录；输出目录必须尚不存在：

```bash
python3 -m scripts.export_bundle /models/Qwen3 /bundles/qwen3-base \
  --verifier /tools/flashengine-native
python3 -m scripts.export_bundle /models/Qwen3 /bundles/qwen3-dflash \
  --draft /models/DFlash --spec-num-tokens 7 \
  --verifier /tools/flashengine-native
```

MoE 使用相同命令，打包器识别 `Qwen3MoeForCausalLM`，归一化专家布局和完整装载计划。k 缺省为草稿 `block_size-1`；这里明确的 3/4 就是 3/4，不采用 Python 历史 CLI sentinel。当前 oracle 工具仍明确拒绝 Python 引擎会误解释的 k=3/4。

完整复制 bundle（包括 `bundle.json`、`engine/`、`weights/`、`tokenizer/`、`golden/`、`provenance/`）到目标机。不要在打包后修改文件，不使用符号链接，不删除失败打包留下的 `INCOMPLETE.json` 来绕过门禁。建议普通文件复制，生产挂载只读；`--hardlink` 仅用于确认原始权重不会被修改的开发场景。

## 3. 公共编译依赖

目标环境需要 CMake >=3.24、支持 C++20 的编译器、OpenSSL、ICU、PCRE2-8、nlohmann/json、Boost >=1.74 头文件和固定 Minja。Ubuntu 22.04 的开发包示例（联网准备阶段）：

```bash
sudo apt-get install build-essential cmake git libssl-dev libicu-dev \
  libpcre2-dev nlohmann-json3-dev libboost-dev
git clone https://github.com/google/minja.git third_party/minja
git -C third_party/minja checkout 021c2293c187789ef13d56c6cfd89c9b134fd80f
cmake --preset host
cmake --build --preset host -j2
ctest --preset host
```

离线机器由软件供应流程预置这些依赖；常规 CMake 不下载任何代码。Minja 可以通过 `-DFE_MINJA_DIR=/opt/deps/minja/include` 指定。`dependencies.lock.json` 目前是开发环境记录，**不是已闭环的正式离线发布锁**。

安装后先执行 `cmake --version`。系统仓库提供的 CMake 不一定达到 3.24，较旧时需要预先准备合格的固定版本；不能只因安装了名为 cmake 的包就认为构建条件满足。

在 AArch64 原生机器运行相同命令即可，不能指向 x86_64 的 `/usr/lib` 或 SDK。初期不建议交叉编译设备后端：sysroot、LibTorch CXX ABI、厂商 runtime 与算子包未闭环，不能以交叉链接成功替代板卡验收。

## 4. CUDA：x86_64 / WSL2 / AArch64

安装与驱动匹配的 CUDA Toolkit 和对应主机架构的 LibTorch C++；准备 FlashInfer 0.6.14 原生头文件。不要把 NVIDIA LibTorch 用于 CoreX。以下变量均需替换为本机路径：

```bash
TORCH_CUDA_ARCH_LIST=8.9 cmake --preset cuda \
  -DCMAKE_PREFIX_PATH=/opt/libtorch \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DFE_DEVICE_TESTS=ON \
  -DFE_FLASHINFER_INCLUDE_DIR=/opt/flashinfer/include
cmake --build --preset cuda -j2
ctest --test-dir build/cuda --output-on-failure
build/cuda/flashengine-native capabilities
```

`89 / 8.9` 是本机 4070 示例，不是跨设备默认：目标卡架构、toolkit 和 FlashInfer 支持面都要重新确认。AArch64 同样在目标机原生编译，并使用 AArch64 LibTorch/CUDA/系统库。WSL2 的 NVIDIA 驱动由 Windows 主机提供，不能把 WSL 库路径当作 Linux/AArch64 发布依赖；建议把编译目录放在 WSL 本地磁盘，避免挂载盘时钟精度造成 make clock-skew 告警。

`cuda-reference` preset 将 CUDA kernels/FlashInfer 替换为 ATen C++ 数学参考实现，用于复验 CoreX 共用算子源码，不证明 CoreX 编译能力，也不提供高性能承诺：

```bash
cmake --preset cuda-reference -DCMAKE_PREFIX_PATH=/opt/libtorch \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build --preset cuda-reference -j2
```

## 5. CoreX：厂商环境上的准备与构建

需要 Iluvatar 驱动、CoreX SDK，以及**提供 ATen/CUDA Graph C++ 接口的厂商 LibTorch 开发包**。仅有 Python `ixformer` wheel 不足以编译此后端。当前没有 SDK 可核对厂商版本、导出符号及 ABI，不能承诺任意 CoreX 版本兼容。

```bash
# 先加载厂商提供的环境脚本，再填写真实路径与 ABI；不能猜 ABI。
cmake --preset corex \
  -DFE_COREX_ROOT=/usr/local/corex \
  -DFE_COREX_TORCH_ROOT=/opt/corex-libtorch \
  -DFE_TORCH_CXX11_ABI=1
cmake --build --preset corex -j2
ctest --test-dir build/corex --output-on-failure
build/corex/flashengine-native capabilities
```

`FE_TORCH_CXX11_ABI=1` 只是占位示例，必须从厂商构建清单确认，0/1 不匹配会造成链接错误或跨库类型损坏。编译器、libstdc++ 和 LibTorch 需成套匹配。源码不编译 NVIDIA PTX，不使用 Triton 或 Python `ixformer`；以厂商预编译 ATen 算子实现固定形状 KV gather / attention / MoE，再经兼容 C++ Graph 捕获。布局是本实现自有 NHD，不复用 Python CoreX 的 HND KV。

该路径是正确性参考，不是 ixformer 高性能 attention 接入。每步会 gather 历史 KV，MoE 会 gather 专家权重；大上下文和大专家模型可能很慢或耗尽内存。若厂商不提供这些 C++/Graph 符号，必须适配原生 SDK，不能引入 Python 来让部署命令表面可运行。AArch64 CoreX 仅在厂商提供对应驱动/LibTorch 时才有构建条件，目前不宣称支持已验证。

## 6. Ascend 310P：AArch64 原生构建

在驱动、固件、CANN 与算子包配套的 310P 容器/主机上加载官方环境脚本，然后构建：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake --preset ascend310p \
  -DFE_ASCEND_ROOT=/usr/local/Ascend/ascend-toolkit/latest
cmake --build --preset ascend310p -j2
ctest --test-dir build/ascend310p --output-on-failure
build/ascend310p/flashengine-native capabilities
```

此 preset 不需要 LibTorch、TorchNPU 或 Python。CMake 检查全部 ACLNN 头文件的调用原型，以及公开 ACL Graph 和 executor 复用/销毁符号；缺失直接报错。版本号本身不够：必须确认这些 API 和所用算子在 **310P** 上可执行，而不是只在 910 上存在。当前未给出已验收的最低 CANN 版本。

实现使用 AscendCL 管理内存/context/stream，ACLNN 两阶段 planner 在图外运行并保留 executor/workspace，ACL Graph 捕获固定执行程序。参考布局为 **ND KV**，不是原方案中待优化的 ATB/NZ 路径；还没有实现 ATB paged attention / NZ 高性能模型组合。它只能作为真机移植起点，不代表已完成 310P 性能方案。算子缺失/不支持、Graph 捕获失败、执行错误均明确失败，不回退 Python 或 TorchNPU 私有符号。

公开 API 依据：[ACL Graph](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1782.html)、[可复用 executor](https://www.hiascend.com/doc_center/source/en/canncommercial/850/API/aolapi/operatorlist_00041.html)。这两份 API 文档不能代替 310P 产品支持和真机结果。

## 7. 启动与 HTTP/SSE

下列 `FE_NATIVE` 按实际后端改成 `build/cuda/flashengine-native`、`build/corex/flashengine-native` 或 `build/ascend310p/flashengine-native`。310P 必须传 `--dtype float16`（本后端也以 fp16 为默认）；CUDA/CoreX 默认 bf16。

```bash
FE_NATIVE=build/cuda/flashengine-native
"$FE_NATIVE" verify /bundles/qwen3-base
"$FE_NATIVE" generate /bundles/qwen3-base --prompt '你好' --chat \
  --max-tokens 32 --max-model-len 2048 --max-num-seqs 4 --kv-blocks 32 --eager
"$FE_NATIVE" serve /bundles/qwen3-dflash \
  --host 127.0.0.1 --port 8000 --served-model-name Qwen3 \
  --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 512 \
  --max-num-partial-prefills 2 --kv-blocks 32 --graph --no-think
```

初次真机验收先 `--eager`，普通路径通过后再 `--graph`，然后 spec eager → spec Graph。CUDA/CoreX Qwen3 捕获 1/2/4/8、16 步进和 `max_num_seqs` 尾桶；Qwen3.5 与新 310P 参考路径的 target 使用固定单行顺序图，DFlash 草稿使用固定块图。任一要求的图失败即拒绝启动。plan、附属状态快照、接受/回滚和上下文提交在图外；上下文尾部走单 token 路径是计划内行为。`--production` 仅禁止 eager 配置，不等于已获 production 资格。

Prefix 和分块预填充均默认开启；`--enable-chunked-prefill` / `--disable-chunked-prefill` 与 Python 对齐。`--max-num-batched-tokens` 是每步 prefill token 预算，`--max-num-partial-prefills` 限制持有状态的 partial prefill；它们计入 max_num_seqs。关闭分块时预算必须覆盖 max_model_len。workload JSON 的 `enable_chunked_prefill` 也会被实际读取。

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3","messages":[{"role":"user","content":"你好"}],"temperature":0,"max_tokens":32,"stream":true}'
curl http://127.0.0.1:8000/v1/stats
curl http://127.0.0.1:8000/metrics
```

设置 `--api-key` 后 `/v1/*` 需要 Bearer key；`/health`、`/metrics` 不鉴权，必须由防火墙/反向代理限制。不要把密钥写入 Git 或公共日志。服务支持 completions、chat completions、SSE；不支持 WebUI、工具调用、随机采样、n>1 或任意 OpenAI 扩展，未支持字段明确返回 400。

## 8. 容量、观测与故障

`--kv-blocks` 为可分配页数，`--kv-block-size` 默认 256；额外 dummy 页不分配给请求。KV 字节近似为：

`2(K/V) × layers × (blocks+1) × block_size × kv_heads × head_dim × dtype_bytes`。

使用 DFlash 时还要加草稿 KV、草稿权重、aux、attention/MoE workspace、各 Graph 桶内存及运行库开销。KV 池至少容纳一条完整 `max_model_len`；并发不足时可能抢占重算。先降低上下文、批次和页数，不要用显存总量减权重大小就把余量全部分给 KV。开发基线为上下文 2048、最大序列 8；8GB 桌面卡应从 1–4 并发小模型开始，不能用它验收 30B MoE 或 4B+训练草稿的真实性能。

启动 stderr 输出完整 resolved config；`/v1/stats`、`/metrics` 共用 host snapshot。检查 `decode_graph_replays`、`draft_graph_replays`、`verify_graph_replays`、KV 使用、抢占和投机接受统计。只有捕获成功但 replay 为零不算 Graph 验收；满接受合成模型也不能证明真实加速。

网络队列有界，慢客户端超限会断开并取消，不阻塞引擎。设备错误或引擎不变量错误退出 70，交给 supervisor 重启；非法请求、鉴权失败、过载和断连局部处理。SIGTERM/SIGINT 停止接入并取消请求，不提供带截止时间的排空。`packaging/flashengine.service` 是 systemd 配置模板，需替换用户、设备权限和路径后使用。

## 9. 真机验收与交付

在 Python 开发环境保存 oracle 和不可覆盖的报告；部署机本身不需要 Python：

```bash
python3 -m scripts.export_oracle /models/Qwen3 /bundles/qwen3-base \
  deployment_output/oracle-new.json --backend cuda --dtype bfloat16 \
  --eager --max-model-len 512 --max-tokens 16
python3 -m scripts.check_native_replay /tools/flashengine-native \
  /bundles/qwen3-base deployment_output/oracle-new.json deployment_output/replay-new.json \
  -- --max-model-len 512 --max-num-seqs 4 --kv-blocks 8 --graph
python3 -m scripts.check_native_server /tools/flashengine-native \
  /bundles/qwen3-base deployment_output/server-new.json \
  -- --max-model-len 256 --max-num-seqs 4 --kv-blocks 8 --graph
python3 -m scripts.native_build_evidence /tools/flashengine-native /src/flashengine-cpp \
  deployment_output/build-new.json
```

310P oracle 改为 `--backend ascend310p --dtype float16`，CoreX 改为 `--backend corex` 并保持相同 dtype。bundle 里包含草稿时 oracle 同时传对应 `--draft`。差分不自动放宽平票；逐轮提案/verify/接受/提交/上下文长度也要一致。`native_build_evidence` 按编译后端采集 nvidia-smi / ixsmi / npu-smi，记录 ELF 架构、动态依赖和源码哈希；工具缺失明确记录，不授予发布资格。

基础算子数值定位可以独立于模型进行。以下只在开发机运行 Python 编译参考，不会向部署程序引入 JIT：

```bash
python3 -m scripts.export_native_ops_oracle deployment_output/ops-new.safetensors \
  --cache-dir /path/to/development-cache
python3 -m scripts.check_native_ops /src/flashengine-cpp/build/cuda/fe-gpu-ops-tests \
  deployment_output/ops-new.safetensors deployment_output/ops-report-new.json
```

测试程序需要 `-DFE_DEVICE_TESTS=ON`。算子参考同时记录源码、Torch 与设备来源；只有实际编译输出才是该 oracle，不能按未编译 Python 源码中的 dtype cast 推断数值。微算子通过仍不等于 attention/GEMM 或整个模型通过。

发布前须完成：batch 1/2/4/8、桶复用与重排、受控接受长度、EOS/max_tokens/上下文截断、抢占取消、慢客户端和故障退出、长稳、热态交替性能 A/B、全部依赖及许可证归档。TPS >= Python 95%、p50 TTFT <=110%，投机 E/逐位置接受率无显著退化。CoreX 与 310P 各自归档报告，不能借用 CUDA 结果。

`packaging/Dockerfile.runtime` 只消费预先准备的原生运行目录和锁定 digest 的厂商运行时镜像，不替你下载 SDK、闭合许可证或完成 release gate。正式 OCI、多架构离线依赖包和 production manifest 尚未验收，不能把示例镜像标签当作正式发布。
