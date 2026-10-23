# 310P1 原生构建与 Qwen3-8B + DFlash 试跑

后续 Graph 严格对比和单轮 benchmark 已完成，见 [Graph 与基准记录](ASCEND310P_GRAPH_BENCHMARK_20261019.md)。以下保留最初 eager 试跑的范围与证据。

2026-10-19，在 `192.168.16.118` 的 AArch64 / Ascend 310P1、`vllm018` 容器中完成构建和单请求 eager 试跑。此记录不授予其他模型组合、Graph、性能或 production 资格；历史失败记录全部保留。

## 部署环境

- 本地源码：`D:/AVIC/FlashEngine.cpp`，提交 `b87b818beea6329f56628673ceff5833ba7b0fa1` 加本轮未提交修正。
- 板端目录：`/data2/FlashEngine.cpp-deploy-b87b818-20261019`，宿主机与容器同路径。
- GCC 12.3.1、CMake 4.3.1、容器 CANN 9.0.T3；链接 AscendCL/ACLNN，没有 LibTorch/TorchNPU/Python 动态依赖。
- Minja 固定提交 `021c2293c187789ef13d56c6cfd89c9b134fd80f`，位于 `third_party/minja`。
- ICU 74.1、PCRE2 10.42 开发包只提取头文件到 `deps/usr/include`，链接容器已有运行库；JSON 头文件为 3.10.5。在线安装超时，没有安装这些系统开发包。
- 板端时钟仍为 2000 年。仅归一化部署副本的源码时间，未改系统时钟；Make 对系统头文件/库仍有 clock-skew 告警。

模型从 `/data/Qwen3-8B`、`/data/Qwen3-8B-Dflash` 普通复制打包到 `bundles/qwen3-8b-dflash`，未修改原模型，未使用硬链接。原生 verify 通过：目标 399 个张量、草稿 58 个张量，契约 0.4.0。

## 修正及验证

1. 修正 Copy、ArgMax、BatchMatMul 的公共头文件名。
2. 普通 BF16/FP16 权重加载增加快速路径，避免逐参数重复构造错误字符串、遍历形状；仍采用既有 FP16 舍入转换。
3. 错误报告增加 ACLNN 算子符号、序号和可用的 ACL 错误详情。
4. 当前 SDK 的 `aclnnInplaceIndexCopy` planner 成功，但 executor 复用返回 `561000`。最小测试复现后，KV 写入改用公开 `aclnnScatterNdUpdate`，保持动态槽位语义。
5. 主机抢占夹具的相同 prompt 在 Prefix 默认开启后消除了预期内存压力，因此仅在该夹具关闭 Prefix。生产 scheduler 未因此改变。

最终 CTest 三项通过：`host-contracts`、`engine-lifecycle`、`ascend-kv-dynamic-slots`。新增设备测试覆盖单行/多行 KV 更新、重复执行中改变索引和数值、未选中行保持；需配置 `-DFE_DEVICE_TESTS=ON`。这些测试不代替完整模型的数值差分。

## 复现

在容器内执行，依赖目录需已按上文准备：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /data2/FlashEngine.cpp-deploy-b87b818-20261019
cmake --preset ascend310p -DFE_DEVICE_TESTS=ON \
  -DFE_ASCEND_ROOT=/usr/local/Ascend/ascend-toolkit/latest \
  -DICU_INCLUDE_DIR="$PWD/deps/usr/include" \
  -DICU_UC_LIBRARY_RELEASE=/usr/lib64/libicuuc.so.74 \
  -DPCRE2_INCLUDE_DIR="$PWD/deps/usr/include" \
  -DPCRE2_LIBRARY=/usr/lib64/libpcre2-8.so.0 \
  -DJSON_INCLUDE_DIR="$PWD/deps/usr/include"
cmake --build --preset ascend310p -j2
ctest --test-dir build/ascend310p --output-on-failure

export TE_PARALLEL_COMPILER=1 CANN_KNOWLEDGE_BANK_PROCESS_NUM=1
build/ascend310p/flashengine-native generate bundles/qwen3-8b-dflash \
  --prompt 'Write a Python function add(a, b) that returns their sum.' \
  --chat --no-think --max-tokens 32 \
  --max-model-len 256 --max-num-seqs 1 --max-num-batched-tokens 64 \
  --kv-blocks 2 --kv-block-size 128 --dtype float16 --eager
```

退出码 0，输出 32 token，包含 `def add(a, b):` 与 `return a + b`。5 轮 DFlash、k=15，原始接受数为 `[7, 2, 0, 15, 15]`。末轮按 32 token 上限截断，不能直接用这些原始接受数计算已输出 token 的接受率。

总墙钟 124.83 秒包含 bundle 校验、加载、算子准备和生成，不是稳态吞吐。未执行 Python oracle 差分、非投机速度对照、长上下文、并发、Prefix 命中或 Graph 验收。CLI 试跑已结束，未留下常驻模型服务。

## 证据

板端 `board-run-evidence.json` 保存最终源码哈希、二进制哈希、bundle manifest 哈希和结果；本地副本位于 `results/ascend310p-20261019/`。

- 二进制 SHA256：`0eefa20405d508d61ddc63643556cead70cadd7b84cfb42de2b05a4bdbaa9d0e`。
- Bundle manifest SHA256：`ea2256db54f865d9be17141c158b72fead382fa14d549d919cf9389823dbf200`。
- 成功运行：`generate-dflash-eager-retry4.log`。
- 最终测试：`ctest-final.log`；构建和设备测试：`build-test-ascend310p-retry4.log`。
- 早期头文件错误、主机断言 #191、慢加载中断、executor 561000，以及一次增量构建未更新执行层对象的记录均保留在旧日志中。

`capabilities` 保持全局 `unvalidated-source-changes` / `production_qualified=false`，本次试跑不扩大为全部能力验收。
