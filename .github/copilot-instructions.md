# Copilot Instructions for AirVLN

## 项目架构与主要组件
- **AirVLN** 是一个面向无人机视觉-语言导航（VLN）任务的研究平台，包含模拟器、模型、数据集和脚本。
- 主要目录：
  - `airsim_plugin/`：AirSim模拟器相关工具，包含客户端和服务端工具（如 `AirVLNSimulatorClientTool.py`、`AirVLNSimulatorServerTool.py`）。
  - `Model/`：导航模型及其训练、推理相关代码（如 `cma_policy.py`、`seq2seq_policy.py`、`il_trainer.py`）。
  - `DATA/`：数据集和预训练模型存放目录。
  - `ENVs/`：环境场景文件夹，包含25个城市级场景。
  - `scripts/`：常用训练、评估、数据收集脚本（如 `train.sh`、`eval.sh`、`collect.sh`）。
  - `utils/`、`src/`：通用工具与核心算法实现。

## 开发者工作流
- **环境准备**：推荐使用 Conda 创建 Python 3.8 虚拟环境，安装依赖见 `requirements.txt` 和 README。
- **模拟器与数据集下载**：需从 Kaggle 下载模拟器和数据集，放置于 `ENVs/` 和 `DATA/data/` 下。
- **模型训练与评估**：
  - 通过 `scripts/train*.sh` 启动训练，参数如 batch size、epoch 可通过脚本名或命令行指定。
  - 评估脚本如 `eval.sh`，需确保模拟器服务已启动。
- **服务启动**：
  - 运行 `AirVLNSimulatorServerTool.py` 启动模拟器服务，默认端口 30000。
  - 客户端通过 `AirVLNSimulatorClientTool.py` 连接服务。
- **常见问题**：
  - 端口占用：如遇 `[Errno 98] Address already in use`，需释放端口或更换端口。
  - 场景加载失败：建议降低 batch size 或检查 GPU 使用情况。

## 项目约定与模式
- **数据流**：指令（自然语言）输入，经模型编码后驱动无人机在模拟环境中导航。
- **模型结构**：主流为 cross-modal-alignment（CMA）和 seq2seq，见 `Model/` 相关文件。
- **环境扩展**：可在 `ENVs/` 新增场景，配置见 `airsim_plugin/settings/`。
- **日志与输出**：训练/评估日志默认输出至工作区根目录，模型输出至 `output/`。

## 关键文件与示例
- `README.md`：安装、数据下载、运行流程详述。
- `airsim_plugin/AirVLNSimulatorClientTool.py`：客户端交互逻辑，注意 `_getImages` 函数的图像通道顺序。
- `scripts/`：批量运行、调试、收集数据的 shell 脚本。

## 依赖与集成
- 依赖 AirSim、PyTorch、pytorch-transformers、Nvidia GPU。
- 与 Habitat 预训练模型兼容（如 `gibson-2plus-resnet50.pth`）。

## 其他说明
- 项目支持 headless/虚拟显示模式，适用于无 GUI 服务器。
- 详细报错与解决方案见 `README.md` 的 Common Errors 部分。

---
如需补充或有疑问，请反馈具体需求或场景。