# FlashEngine Native（candidate）

独立维护的 C++20 部署实现，Python FlashEngine 负责实验与 oracle。本目录是独立 Git 仓，父仓忽略它；尚未创建远程仓库或提交。当前实现不是生产版本，不能沿用 Python 仓的支持声明。

## 已实现与未验收

已实现 Linux 原生 bundle/safetensors loader、Qwen byte-level BPE（PCRE2 + ICU NFC）、Minja chat template、主机调度器、Paged KV、chunked prefill、抢占/取消、Qwen3 dense/MoE CUDA eager/Graph、DFlash1 多图路径以及 Boost.Beast HTTP/SSE。MoE 支持逐专家和融合 3D 权重、mixed dense/sparse 层及 top-k 路由。

CoreX 厂商 LibTorch C++ 后端和 AscendCL/ACLNN/ACL Graph 后端已准备源码及 CMake 接口，但没有 SDK/真机，尚未编译验收。CoreX 使用固定形状 ATen 参考 attention，310P 使用 ACLNN ND 参考 attention；ixformer 原生高性能路径、ATB/NZ 优化仍未完成。实测范围和失败项见 [验收记录](docs/VALIDATION.md)。DFlash 的最终 token 一致不等于逐轮 trace 通过；合成 MoE 的严格 trace 通过也不替代真实模型验收。没有任何后端获得 production 资格。

完整的 **x86_64 / WSL2 / AArch64、CUDA / CoreX / 310P** 构建、启动、容量预算和运维说明见 [部署手册](docs/DEPLOYMENT.md)。正式 OCI 镜像、完整离线依赖发布包尚未验收；`packaging/` 仅提供运行镜像和 systemd 模板。

原生 `benchmark` 命令、Python/C++ ABBA 口径及已知数值分歧见 [基准说明](docs/BENCHMARK.md)。严格正确性失败时，性能达标也不能放行。

## 构建

仅支持 Linux；Windows 从 WSL 构建。需要 CMake >=3.24、C++20、OpenSSL、ICU、PCRE2、nlohmann/json、Boost 头文件。版本见 `dependencies.lock.json`，当前锁文件记录开发环境，不是完整可重现发布锁。

Minja 必须预先准备，常规 CMake 构建不联网：

```bash
git clone https://github.com/google/minja.git third_party/minja
git -C third_party/minja checkout 021c2293c187789ef13d56c6cfd89c9b134fd80f
cmake --preset host
cmake --build --preset host -j2
ctest --preset host
cmake --preset host-asan
cmake --build --preset host-asan -j2
ctest --preset host-asan
# 可选：仅测试主机模型的真实网络服务，配合 Python check_native_shutdown 使用
cmake --build build/asan --target fe-server-probe -j2
```

CUDA 需要与驱动匹配的 LibTorch C++ 与 CUDA toolkit、FlashInfer 0.6.14 原生头文件；本机使用 Torch 2.11.0+cu130、CUDA 13.0、sm_89。`FE_FLASHINFER_INCLUDE_DIR` 指向包含 `flashinfer/attention/prefill.cuh` 的目录：

```bash
cmake --preset cuda \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DFE_FLASHINFER_INCLUDE_DIR=/path/to/flashinfer/include
cmake --build --preset cuda -j2
ctest --test-dir build/cuda --output-on-failure
```

CUDA kernel 是预编译代码，不调用 Python、Triton 或 Python JIT。开发构建借用了现有 Python 环境内的 Torch C++ 库；运行时依赖以 `ldd` 证据为准。完整独立发布镜像及全部许可证归档仍待完成。FlashInfer planner 的 CUDA 13.0 fastdiv 兼容层使用普通整数除法，性能必须另测。

## 运行

部署 bundle 由 Python 仓的 `scripts.export_bundle` 生成；`verify` 校验哈希、权重计划和 tokenizer/chat golden，失败拒绝启动。当前 CUDA/CoreX 仅编译 head_dim=128、TP=1、Qwen3 dense/MoE、fp16/bf16、greedy；310P 仅 fp16。DFlash1 只接受全词表、非因果全注意力。

```bash
build/cuda/flashengine-native --help
build/cuda/flashengine-native capabilities
build/cuda/flashengine-native verify /path/to/bundle
build/cuda/flashengine-native generate /path/to/bundle \
  --prompt '你好' --chat --max-tokens 32 \
  --max-model-len 2048 --max-num-seqs 4 --kv-blocks 32 --graph
build/cuda/flashengine-native serve /path/to/bundle \
  --host 127.0.0.1 --port 8000 --served-model-name Qwen3 \
  --max-model-len 2048 --max-num-seqs 4 --kv-blocks 32 --graph --no-think
build/cuda/flashengine-native replay /path/to/bundle \
  --oracle /path/to/oracle.json --max-model-len 512 \
  --max-num-seqs 1 --kv-blocks 8 --eager --logits
```

`--eager` 是开发默认；`--graph` 在启动时捕获全部配置桶（1/2/4/8、16 步进及尾桶），失败拒绝启动，不静默降级。`--production` 只是严格 Graph 配置，不授予发布资格。KV 池必须至少容纳一条完整上下文；额外一页仅供图 padding，不分配给请求。

接口：`/health`、`/v1/models`、`/v1/completions`、`/v1/chat/completions`、`/v1/stats`、`/metrics`（别名 `/v1/metrics`）。设置 `--api-key` 后 `/v1/*` 要求 Bearer key；`/health`、`/metrics` 不鉴权，必须通过网络边界保护。未知字段、非 greedy、`n!=1` 明确 400。没有 WebUI。

网络线程不驱动模型；每连接发送队列 256 KiB、驱动到网络邮箱 64 KiB，溢出关闭连接并取消。请求队列有界，满返回 429。主机 snapshot 不发起设备同步。设备/引擎异常退出 70；SIGTERM/SIGINT 停止接入并取消剩余请求。尚未实现有期限的优雅排空。

## 维护

契约当前为 0.2.0，`contracts.lock.directory` 指向 `vendor/contracts-0.2.0`；保留旧 0.1 快照及历史 lock，可读取 0.1 dense bundle。快照只能通过 Python 发布工具更新；主机测试校验哈希。源码分层为 `engine/scheduler → DeviceModel → cuda_model/ascend_model → 原生算子`，CLI/HTTP 和公共调度不包含 ATen/CUDA/ACL 类型。

流程保持：Python 实验 → 契约冻结 → golden → C++ 转译 → 每后端独立验收。不要把仅能编译、仅最终 token 相同、仅成功捕获 Graph，写成阶段验收完成。
