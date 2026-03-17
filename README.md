# OmniVideo-Analyst

🎬 **Ask anything about your video.**

基于 Qwen3-Omni 多模态大模型的视频智能分析工具。使用自然语言查询视频内容，快速定位关键片段。

---

## ✨ 特性

- 🔍 **自然语言查询** - 用日常语言描述你想找的内容，如"红色自行车"、"门有没有打开过"
- 🎯 **精准时间定位** - 自动输出相关片段的时间戳
- 📹 **多格式支持** - 支持 MP4、AVI、MOV、MKV 等常见视频格式
- 🚀 **高性能推理** - 支持多 GPU 张量并行，加速长视频分析
- 🔒 **本地部署** - 数据不出本地，适合敏感场景（警务、企业、教育等）

---

## 🎯 应用场景

| 场景 | 示例查询 |
|------|---------|
| **安防监控** | "过去 1 小时内有没有人翻越围墙？" |
| **零售分析** | "今天有多少顾客穿了红色衣服？" |
| **教育教学** | "找出老师讲解三角函数的所有片段" |
| **体育赛事** | "分析这场比赛的所有进球瞬间" |
| **内容审核** | "检测视频中是否出现违规内容" |
| **个人视频** | "找我去年生日吹蜡烛的片段" |

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                   用户接口层                              │
│         CLI / Web UI / API (FastAPI)                     │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  核心分析引擎                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ 视频预处理   │  │ 大模型推理   │  │ 结果聚合输出     │  │
│  │  - 切段      │  │  - Qwen3   │  │  - 时间轴        │  │
│  │  - 关键帧    │  │  - Omnip   │  │  - JSON 报告      │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              推理后端 (vLLM / Transformers)               │
│         Qwen3-Omni-30B-A3B (量化版本)                     │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 安装

### 环境要求

- Python 3.10+
- GPU: 推荐 NVIDIA 显卡 (显存 ≥ 24GB，支持多卡并行)
- CUDA 12.0+

### 快速开始

```bash
# 克隆仓库
git clone https://github.com/XHJ-Studio/OmniVideo-Analyst.git
cd OmniVideo-Analyst

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 下载模型（首次运行自动下载）
# 或使用 ModelScope（中国大陆推荐）
modelscope download --model Qwen/Qwen3-Omni-30B-A3B-Thinking
```

---

## 🚀 使用示例

### CLI 模式

```bash
# 分析视频，查询特定内容
python src/main.py \
    --video path/to/video.mp4 \
    --query "视频里有没有人打开过门？" \
    --output report.json

# 使用多 GPU 加速
python src/main.py \
    --video path/to/video.mp4 \
    --query "找出一辆红色自行车" \
    --tensor-parallel-size 3 \
    --output report.json
```

### API 模式

```bash
# 启动 API 服务
python src/api.py --port 8000

# 调用 API
curl -X POST http://localhost:8000/analyze \
    -H "Content-Type: application/json" \
    -d '{
        "video_path": "path/to/video.mp4",
        "query": "有没有人翻越围墙？"
    }'
```

### Python SDK

```python
from omnivideo import VideoAnalyst

# 初始化分析器
analyst = VideoAnalyst(
    model_path="Qwen/Qwen3-Omni-30B-A3B-Thinking",
    tensor_parallel_size=3
)

# 分析视频
result = analyst.analyze(
    video_path="surveillance.mp4",
    query="过去 10 分钟内有没有可疑人员出现？"
)

# 输出结果
print(f"发现 {len(result.segments)} 个相关片段")
for seg in result.segments:
    print(f"  [{seg.start_time} - {seg.end_time}] {seg.description}")
```

---

## 📊 性能参考

| 视频长度 | 切段策略 | GPU 配置 | 分析耗时 |
|---------|---------|---------|---------|
| 5 分钟 | 60 秒/段 | 1×RTX 4090 | ~3 分钟 |
| 15 分钟 | 60 秒/段 | 3×RTX 2080Ti | ~8 分钟 |
| 1 小时 | 60 秒/段 | 3×RTX 2080Ti | ~30 分钟 |

*实际耗时取决于查询复杂度和视频内容*

---

## 📁 项目结构

```
OmniVideo-Analyst/
├── README.md                 # 项目说明
├── LICENSE                   # Apache 2.0 许可证
├── requirements.txt          # Python 依赖
├── .gitignore               # Git 忽略规则
├── src/
│   ├── main.py              # CLI 入口
│   ├── api.py               # API 服务
│   ├── analyst.py           # 核心分析逻辑
│   ├── preprocessor.py      # 视频预处理
│   └── utils/
│       ├── prompt.py        # Prompt 模板
│       └── output.py        # 结果格式化
├── tests/                   # 测试用例
├── configs/                 # 配置文件
├── examples/                # 示例视频和查询
└── docs/                    # 文档
```

---

## 🔐 安全与隐私

- ✅ 所有视频数据本地处理，不上传云端
- ✅ 模型权重本地存储
- ✅ 支持内网部署
- ✅ 适合敏感场景（警务、金融、医疗等）

---

## 📄 许可证

Apache License 2.0 - 允许商业使用，保留专利权利

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📬 联系方式

- GitHub: [XHJ-Studio](https://github.com/XHJ-Studio)
- 项目仓库：[OmniVideo-Analyst](https://github.com/XHJ-Studio/OmniVideo-Analyst)

---

<div align="center">

**Made with ❤️ by 小黄鸡工坊**

[⭐ Star this repo](https://github.com/XHJ-Studio/OmniVideo-Analyst) if you find it useful!

</div>
