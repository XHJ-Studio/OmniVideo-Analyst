# OmniVideo-Analyst 使用示例

本目录包含各种使用示例，帮助你快速上手 OmniVideo-Analyst。

---

## 📋 示例列表

| 示例 | 说明 | 适用场景 |
|------|------|---------|
| `basic_analysis.py` | 基础视频分析 | 快速测试 |
| `batch_analysis.py` | 批量视频分析 | 多视频处理 |
| `api_client.py` | API 客户端示例 | 集成到现有系统 |
| `websocket_client.py` | WebSocket 实时进度 | 前端集成 |
| `custom_prompt.py` | 自定义 Prompt 模板 | 特定场景优化 |

---

## 1. 基础视频分析

```bash
# 运行示例
python examples/basic_analysis.py --video path/to/video.mp4 --query "有没有人翻越围墙？"
```

**功能：**
- 加载视频
- 自动切段
- 分析并输出结果
- 保存 JSON 报告

---

## 2. 批量视频分析

```bash
# 运行示例
python examples/batch_analysis.py --input-dir ./videos/ --query "找红色汽车" --output-dir ./reports/
```

**功能：**
- 批量处理多个视频
- 并行分析
- 汇总报告

---

## 3. API 客户端

```bash
# 启动 API 服务
python src/api.py --port 8000

# 运行客户端
python examples/api_client.py --video path/to/video.mp4 --query "门有没有打开过"
```

**功能：**
- 提交分析任务
- 轮询任务状态
- 获取分析结果

---

## 4. WebSocket 实时进度

```bash
# 运行 WebSocket 客户端
python examples/websocket_client.py --task-id <task_id>
```

**功能：**
- 实时接收进度更新
- 显示分析状态
- 接收完成通知

---

## 5. 自定义 Prompt

```bash
# 运行示例
python examples/custom_prompt.py --video path/to/video.mp4 --template suspicious
```

**可用模板：**
- `general` - 通用分析
- `door_window` - 门/窗状态检测
- `person` - 人员检测
- `vehicle` - 车辆检测
- `suspicious` - 可疑行为检测
- `object` - 特定物体搜索

---

## 📖 详细文档

- [README.md](../README.md) - 项目介绍
- [docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md) - 开发指南
- [docs/PROGRESS.md](../docs/PROGRESS.md) - 开发进度

---

<div align="center">

**Made with ❤️ by 小黄鸡工坊**

[⭐ GitHub](https://github.com/XHJ-Studio/OmniVideo-Analyst)

</div>
