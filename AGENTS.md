# FlashEngine C++ 维护约定

- 本目录是独立 Git 仓；Python 父仓不跟踪这里的内容。
- 注释、文档和命令行帮助使用中文。
- Python 是行为 oracle，契约通过 vendor/contracts 和 contracts.lock 固定；禁止人工修改快照。
- Engine/Scheduler 不依赖 ATen、CUDA、ACL。设备差异留在后端。
- 部署运行时不得加载 Python、Rust 或 Python JIT。构建和离线 oracle 工具可在 Python 仓运行。
- 仅将真实验收通过的能力标为 validated；310P 没有真机报告不得标为 production。
- 非预期引擎错误 fail-fast；客户端校验、过载和断连局部处理。
- 主机测试：cmake --build build/host；ctest --test-dir build/host --output-on-failure。
- CUDA 改动必须实际运行 replay；Graph 必须包含非零 replay 计数和动态上下文回归。
