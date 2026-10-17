# FlashEngine C++ 维护约定

**2026-10-17 三后端源码对齐**：本轮用户明确要求只改代码、不测试，未构建、未运行任何测试，现有 build 仍为旧程序。310P 新增纯 ACLNN Qwen3.5 dense/MoE、DFlash 回滚、EAGLE3 shifted/carry/私有 KV/Prefix，以及 Quark 加载期反量化；target 使用单行顺序 Graph，ND 参考 attention，不是 ATB/NZ/真 int8。CoreX 继续共用 CUDA 模型层。公共 scheduler 补齐 chunked-prefill CLI/workload 开关和统计；详见 `docs/BACKEND_PARITY_SOURCE.md`。所有本轮修改均 unvalidated-source-changes；下方历史通过报告不覆盖它们。不得删除失败记录或把源码对齐标为编译/行为/性能验收。

**0.4 组合增量**：Prefix 默认开启；Qwen3 dense/MoE 可组合 DFlash/EAGLE3，Qwen3.5 文本 dense/MoE 可组合 DFlash。附属状态须通过 prefix_ready 门控；Qwen3.5 用 CPU GDN 边界快照与逐验证行回滚，EAGLE3 用独立草稿 KV+aux 重建。Graph 为参考分图/单行顺序验证，不宣称融合 verify。最新范围见 `docs/FEATURE_COMBINATIONS.md`；验收只认 `docs/VALIDATION.md` 的实际报告。以下 0.3 条目是历史记录；真实 EAGLE3 逐轮差异和后端真机门禁仍未关闭。

**0.3 增量（2026-10-17）**：新增 Qwen3 普通路径 Prefix、EAGLE3、Qwen3.5 文本参考执行与单行 Graph、Quark int8 加载期反量化。不能将加载期转换称为 int8 GEMM。真实 EAGLE3 逐轮差分仍失败；Qwen3.5 不支持投机回滚或 Prefix。能力、失败记录与复现分别见 `docs/FEATURES_0_3.md`、`docs/VALIDATION.md`。CoreX 未编译/未上板，310P 拒绝新增的 EAGLE3/Qwen3.5/Quark profile；所有后端仍 candidate。

- 本目录是独立 Git 仓；Python 父仓不跟踪这里的内容。
- 注释、文档和命令行帮助使用中文。
- Python 是行为 oracle，契约通过 vendor/contracts 和 contracts.lock 固定；禁止人工修改快照。
- Engine/Scheduler 不依赖 ATen、CUDA、ACL。设备差异留在后端。
- 部署运行时不得加载 Python、Rust 或 Python JIT。构建和离线 oracle 工具可在 Python 仓运行。
- 仅将真实验收通过的能力标为 validated；310P 没有真机报告不得标为 production。
- 非预期引擎错误 fail-fast；客户端校验、过载和断连局部处理。
- 主机测试：cmake --build build/host；ctest --test-dir build/host --output-on-failure。
- CUDA 改动必须实际运行 replay；Graph 必须包含非零 replay 计数和动态上下文回归。
