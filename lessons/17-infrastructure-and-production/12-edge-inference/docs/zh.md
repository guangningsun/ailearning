# 边缘推理——Apple Neural Engine、高通 Hexagon、WebGPU/WebLLM、Jetson

> 边缘的核心约束是内存带宽，不是算力。移动 DRAM 为 50-90 GB/s；数据中心 HBM3 高达 2-3 TB/s——差距 30-50 倍。解码是内存绑定的，所以这个差距是决定性的。2026 年格局分为四路。Apple M4/A18 Neural Engine 峰值 38 TOPS，采用统一内存（无 CPU↔NPU 拷贝）。高通 Snapdragon X Elite / 8 Gen 4 Hexagon 达 45 TOPS。WebGPU + WebLLM 在 M3 Max 上运行 Llama 3.1 8B Q4 约 41 tok/s（约 70-80% 原生性能）；GitHub 17.6k 星，OpenAI 兼容 API，约 70-75% 移动端覆盖。NVIDIA Jetson Orin Nano Super（8GB）可运行 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 运行 gpt-oss-20b 约 40 tok/s；Jetson T4000（JetPack 7.1）是 AGX Orin 的 2 倍性能。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4、分块 prefill——在 CES 2026 上由 Bosch、ThunderSoft、MediaTek 展示。

**类型：** 学习型
**语言：** Python（标准库，含 toy 带宽受限解码模拟器）
**前置条件：** 阶段 17 · 04（vLLM 服务内幕）、阶段 17 · 09（生产量化）
**时间：** 约 60 分钟

## 学习目标

- 解释为什么移动 LLM 推理是内存带宽受限的，算力是次要的。
- 列举四个边缘目标（Apple ANE、高通 Hexagon、WebGPU/WebLLM、NVIDIA Jetson）并为每个匹配一个用例。
- 说出 2026 年 WebGPU 覆盖缺口（Firefox Android 正在追赶）和 Safari iOS 26 落地情况。
- 为每个目标选择量化格式（ANE 用 Core ML INT4 + FP16，Hexagon 用 QNN INT8/INT4，浏览器用 WebGPU Q4，Jetson 用 NVFP4）。

## 问题

客户想要一个设备端聊天机器人：语音优先、默认私密、离线工作。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 运行约 55 tok/s——可以接受。在 iPhone 16 Pro 上，同样的模型运行 3 tok/s——不行。在搭载 Snapdragon 8 Gen 3 的中端 Android 上，7 tok/s。通过 Chrome Android v121+ 上的 WebGPU 在浏览器中，4-8 tok/s，取决于设备。

吞吐量差异不是移植问题。是带宽差距乘以量化格式再乘以 NPU 是否可从用户空间访问。2026 年边缘推理是四个不同问题，需要四种不同解决方案。

## 概念

### 带宽才是真正的天花板

解码为每个 token 读取完整权重集。一个 Q4 的 7B 模型是 3.5 GB。以 50 GB/s 读取 3.5 GB 需要 70 ms——理论上限约 14 tok/s。以 90 GB/s（高端移动 DRAM）计算，上限升至约 25 tok/s。在这个数字以下，多少算力都没用。

数据中心 HBM3 以 3 TB/s 清除同样的 3.5 GB 只需 1.2 ms——上限是 830 tok/s。同样的模型，同样的权重。不同的内存子系统。

### Apple Neural Engine（M4 / A18）

- 高达 38 TOPS。统一内存（CPU 和 ANE 共享同一池）——无拷贝开销。
- 通过 Core ML + `.mlmodel` 编译模型访问，或通过 PyTorch 的 Metal Performance Shaders（MPS）。
- Llama.cpp Metal 后端使用 MPS，不直接使用 ANE；原生 ANE 需要 Core ML 转换。
- 2026 年 iOS 应用的最佳实践路径：Core ML + INT4 权重 + FP16 激活。

### 高通 Hexagon（Snapdragon X Elite / 8 Gen 4）

- 高达 45 TOPS。与 CPU 和 GPU 集成在 SoC 中但内存域分离。
- QNN（Qualcomm Neural Network）SDK 和 AI Hub 提供从 PyTorch/ONNX 的转换。
- 聊天模板、Llama 3.2、Phi-3 都在 AI Hub 上作为一级制品提供。

### Intel / AMD NPU（Lunar Lake、Ryzen AI 300）

- 40-50 TOPS。软件落后于 Apple/Qualcomm；OpenVINO 在改进但仍小众。
- 最适合 Windows ARM copilot 应用；在 AMD/Intel 台式机上用于本地优先。

### WebGPU + WebLLM

- 通过 WebGPU 计算着色器在浏览器中运行模型；无需安装。
- Llama 3.1 8B Q4 在 M3 Max 上约 41 tok/s——通过同一后端约 70-80% 原生性能。
- WebLLM 在 GitHub 有 17.6k 星；OpenAI 兼容 JS API；Apache 2.0。
- 2026 年覆盖：Chrome Android v121+、Safari iOS 26 GA、Firefox Android 仍在追赶。总体约 70-75% 移动端覆盖。

### NVIDIA Jetson 系列

- Orin Nano Super（8GB）：可运行 Llama 3.2 3B、Phi-3 tok/s 表现良好。
- AGX Orin：通过 vLLM 运行 gpt-oss-20b 约 40 tok/s。
- Thor / T4000（JetPack 7.1）：AGX Orin 2 倍性能，支持 EAGLE-3 和 NVFP4。
- TensorRT Edge-LLM（2026）支持 EAGLE-3 投机解码、NVFP4 权重、分块 prefill——数据中心优化移植到边缘。

### 各目标量化选择

| 目标 | 格式 | 说明 |
|--------|--------|-------|
| Apple ANE | INT4 权重 + FP16 激活 | Core ML 转换路径 |
| 高通 Hexagon | QNN INT8 / INT4 | AI Hub 转换器 |
| WebGPU / WebLLM | Q4 MLC（q4f16_1） | 使用 `mlc_llm convert_weight` + 编译 `.wasm`；不支持 GGUF |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 内存受限 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路径 |

### 边缘上的长上下文陷阱

Llama 3.1 的 128K 上下文是数据中心特性。在 8 GB RAM 的手机上，4 GB 模型 + 2 GB 32K token KV 缓存 + 系统开销 = OOM。边缘部署将上下文保持在 4K-8K，除非接受激进的 KV 量化（Q4 KV）。

### 语音是杀手级应用

语音代理对延迟敏感（首个 token < 500 ms）。本地推理完全消除网络延迟。结合语音转文字（Whisper Turbo 变体在边缘运行），边缘推理成为生产级语音循环。

### 应记住的数字

- Apple M4 / A18 ANE：38 TOPS。
- 高通 Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4 约 41 tok/s。
- AGX Orin：通过 vLLM 运行 gpt-oss-20b 约 40 tok/s。
- 数据中心-边缘带宽差距：30-50 倍。
- WebGPU 移动端覆盖：约 70-75%（Firefox Android 落后）。

## 使用方法

`code/main.py` 根据边缘目标的带宽受限数学计算理论解码吞吐量上限。与观察到的基准测试比较，并突出显示带宽而非算力是瓶颈的地方。

## 交付

本课产出 `outputs/skill-edge-target-picker.md`。给定平台（iOS/Android/browser/Jetson）、模型和延迟/内存预算，选择量化格式和转换流水线。

## 练习

1. 运行 `code/main.py`。对于在 Snapdragon 8 Gen 3（约 77 GB/s 带宽）上 Q4 的 7B 模型，计算解码上限。与观察到的 6-8 tok/s 比较——运行时高效吗？
2. Android 上的 WebGPU 需要 Chrome v121+。为旧版浏览器设计回退方案——通过相同的 OpenAI 兼容 API 在服务端。
3. 你的 iOS 应用需要 4K 上下文流式处理。哪种模型/格式组合能让你在 iPhone 16 上保持在 4 GB 活跃内存以下？
4. Jetson AGX Orin 运行 gpt-oss-20b 达 40 tok/s。Jetson Nano 只能运行 3B。如果你的产品同时面向两者，如何统一推理栈？
5. 论证"WebLLM 在 2026 年已具备生产就绪性"。引用覆盖、性能和 Firefox Android 差距。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| ANE | "Apple 神经引擎" | M 系列和 A 系列上的设备端 NPU；统一内存 |
| Hexagon | "高通 NPU" | Snapdragon NPU；通过 QNN SDK 访问 |
| WebGPU | "浏览器 GPU" | W3C 标准化的浏览器 GPU API；Chrome/Safari 2026 |
| WebLLM | "浏览器 LLM 运行时" | MLC-LLM 项目；Apache 2.0；OpenAI 兼容 JS |
| Jetson | "NVIDIA 边缘" | Orin Nano / AGX / Thor / T4000 系列 |
| TRT Edge-LLM | "边缘 TensorRT" | 2026 年 TensorRT-LLM 边缘移植；EAGLE-3 + NVFP4 |
| 统一内存 | "共享池" | CPU 和 NPU 看到同一 RAM；无拷贝开销 |
| 带宽受限 | "内存受限" | 解码受读取权重的字节/秒限制 |
| Core ML | "Apple 转换" | Apple 框架，用于 ANE 原生模型 |
| QNN | "高通栈" | Qualcomm Neural Network SDK |

## 扩展阅读

- [设备端 LLM 现状 2026](https://v-chandra.github.io/on-device-llms/) — 格局和基准测试。
- [NVIDIA Jetson 边缘 AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 年边缘移植公告。
- [WebLLM（arXiv:2412.15803）](https://arxiv.org/html/2412.15803v2) — 设计和基准测试。
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE 原生转换。
- [高通 AI Hub](https://aihub.qualcomm.com/) — 为 Hexagon 预转换的模型。
