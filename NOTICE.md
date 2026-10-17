# 来源与依赖说明

本实现转译 FlashEngine Python 引擎的行为契约，保留其 MIT 许可。架构参考 localai-org/vllm.cpp；当前自有引擎代码没有复制其 tensor runtime。

- FlashInfer 原生 C++/CUDA 头文件：Apache-2.0，构建时提供，必须保留上游 LICENSE/NOTICE。CUDA 13.0 的整数除法兼容层位于自有源码，不修改上游安装。
- Minja：MIT，固定提交 021c2293c187789ef13d56c6cfd89c9b134fd80f；依赖准备后保留仓库 LICENSE。
- LibTorch：BSD-style，发布时保留官方 LICENSE 与第三方声明，不链接 libtorch_python。
- Boost：Boost Software License 1.0。
- nlohmann/json：MIT。
- PCRE2：BSD-style；ICU：Unicode/ICU License；OpenSSL：Apache-2.0（3.x）。

系统库和编译器的实际版本记录在 dependencies.lock.json。发布前必须收集所有动态依赖的许可与文件哈希。
