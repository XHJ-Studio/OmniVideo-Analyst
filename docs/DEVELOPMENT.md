# OmniVideo-Analyst 开发指南

**版本**: 0.1.0  
**最后更新**: 2026-03-18  
**维护者**: 小黄鸡工坊

---

## 🎯 项目目标

开发一个基于 Qwen3-Omni 多模态大模型的视频智能分析工具，支持：
- 自然语言视频查询
- 精准时间定位
- 多场景应用（安防、零售、教育、体育等）
- 本地化部署（数据隐私保护）

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                   用户接口层                              │
│         CLI (main.py) / API (api.py) / Web UI            │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  核心分析引擎                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ 预处理      │  │ 推理引擎    │  │ 结果聚合        │  │
│  │ preprocessor│  │ inference   │  │ aggregator      │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  模型层                                  │
│         model_loader.py (vLLM / Transformers)            │
│         Qwen3-Omni-30B-A3B (量化版本)                     │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 模块说明

### 1. 视频预处理模块 (src/preprocessor.py)

**功能**:
- 视频文件加载与验证
- 视频切段（按时间长度）
- 关键帧提取（场景变化检测）
- 音频轨道提取
- 格式转换（统一编码）

**输入**:
- 视频文件路径
- 切段长度（秒）
- 输出目录

**输出**:
- 视频片段列表
- 关键帧图像
- 音频文件（可选）

**依赖**:
- opencv-python
- ffmpeg-python
- av

**测试方法**:
```bash
python tests/test_preprocessor.py
```

---

### 2. 模型加载模块 (src/model_loader.py)

**功能**:
- 模型下载（ModelScope / HuggingFace）
- 模型加载（vLLM / Transformers）
- 多 GPU 张量并行配置
- 显存优化（量化、offload）

**输入**:
- 模型名称或路径
- GPU 数量
- 精度（FP16/INT8/INT4）

**输出**:
- 加载的模型实例
- Processor 实例

**依赖**:
- vllm (qwen3_omni 分支)
- transformers
- accelerate
- modelscope

**测试方法**:
```bash
python tests/test_model_loader.py
```

---

### 3. 推理引擎 (src/inference.py)

**功能**:
- 单段视频分析
- Prompt 构建与优化
- 多模态输入处理（视频 + 音频 + 文本）
- 结果解析（JSON 提取）

**输入**:
- 视频片段路径
- 查询文本
- Prompt 模板

**输出**:
- 分析结果（字典）
- 置信度评分

**依赖**:
- model_loader 模块
- utils/prompt 模块

**测试方法**:
```bash
python tests/test_inference.py
```

---

### 4. 结果聚合模块 (src/aggregator.py)

**功能**:
- 多段结果合并
- 时间轴生成
- 置信度计算与过滤
- JSON/Markdown 报告输出

**输入**:
- 多段分析结果列表
- 视频总时长
- 查询文本

**输出**:
- 聚合报告（JSON/Markdown）
- 时间轴可视化数据

**依赖**:
- 无外部依赖

**测试方法**:
```bash
python tests/test_aggregator.py
```

---

## 🚀 开发环境配置

### 系统要求

- Python 3.10+
- GPU: NVIDIA 显卡（显存 ≥ 24GB，推荐多卡）
- CUDA 12.0+
- 磁盘空间：≥ 100GB（模型权重 + 缓存）

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/XHJ-Studio/OmniVideo-Analyst.git
cd OmniVideo-Analyst

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装 vLLM（Qwen3-Omni 分支）
git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git
cd vllm
pip install -e .

# 5. 验证安装
python -c "from src.analyst import VideoAnalyst; print('OK')"
```

---

## 📝 开发规范

### Git 提交规范

```bash
git commit -m "类型：模块 - 功能描述

- 新增：xxx
- 修改：xxx  
- 修复：xxx

下一步：xxx
"
```

**类型**:
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具

### 代码规范

1. 使用 type hints（Python 3.10+ 语法）
2. 每个函数必须有 docstring
3. 错误必须捕获并记录日志
4. 关键步骤必须有进度提示

### 测试规范

1. 每个模块必须有对应的测试文件
2. 测试覆盖率 ≥ 80%
3. 使用 pytest 框架
4. 测试数据放在 tests/data/

---

## 🔧 常见问题

### Q1: 显存不足怎么办？

**A**: 
1. 使用量化版本模型（AWQ-8bit/4bit）
2. 减少 tensor_parallel_size
3. 启用 CPU offload
4. 减小 batch_size

### Q2: 视频解码失败？

**A**:
1. 检查 ffmpeg 是否安装：`ffmpeg -version`
2. 安装系统级解码库：`apt install libavcodec-dev`
3. 尝试转换视频格式：`ffmpeg -i input.mp4 -c:v libx264 output.mp4`

### Q3: 模型下载太慢？

**A**:
1. 使用 ModelScope（中国大陆）：`modelscope download --model Qwen/...`
2. 使用镜像站：`hf-mirror.com`
3. 手动下载后放到本地路径

---

## 📚 参考资料

- [Qwen3-Omni 官方文档](https://github.com/QwenLM/Qwen3-Omni)
- [vLLM 文档](https://docs.vllm.ai/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [OpenCV 文档](https://docs.opencv.org/)

---

## 🤝 贡献指南

1. Fork 仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m "feat: add your feature"`
4. 推送到分支：`git push origin feature/your-feature`
5. 提交 Pull Request

---

## 📬 联系方式

- GitHub Issues: https://github.com/XHJ-Studio/OmniVideo-Analyst/issues
- 邮箱：（待添加）

---

<div align="center">

**Made with ❤️ by 小黄鸡工坊**

Apache License 2.0

</div>
