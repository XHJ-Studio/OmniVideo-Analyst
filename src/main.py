#!/usr/bin/env python3
"""
OmniVideo-Analyst CLI 入口

使用示例:
    python src/main.py --video path/to/video.mp4 --query "视频里有没有人打开过门？"
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="OmniVideo-Analyst - 使用自然语言分析视频内容",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --video surveillance.mp4 --query "有没有人翻越围墙？"
  %(prog)s --video meeting.mp4 --query "找出所有提到预算的片段" --output report.json
  %(prog)s --video traffic.mp4 --query "红色汽车" --segment-length 30
        """
    )
    
    parser.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="输入视频文件路径"
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="自然语言查询，如'找红色自行车'、'门有没有打开过'"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出报告文件路径 (JSON 格式)，默认输出到控制台"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Thinking",
        help="模型名称或本地路径 (默认：Qwen/Qwen3-Omni-30B-A3B-Thinking)"
    )
    
    parser.add_argument(
        "--segment-length",
        type=int,
        default=60,
        help="视频切段长度 (秒)，默认 60 秒"
    )
    
    parser.add_argument(
        "--tensor-parallel-size", "-tp",
        type=int,
        default=1,
        help="张量并行 GPU 数量，默认 1"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="推理设备 (cuda/cpu)，默认 cuda"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细输出"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 验证输入文件
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ 错误：视频文件不存在：{video_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"🎬 OmniVideo-Analyst v{__import__('src').__version__}")
    print(f"=" * 50)
    print(f"📹 视频：{video_path.name}")
    print(f"🔍 查询：{args.query}")
    print(f"🧠 模型：{args.model}")
    print(f"📏 切段：{args.segment_length}秒/段")
    print(f"🎮 GPU: {args.tensor_parallel_size}卡并行")
    print(f"=" * 50)
    
    # TODO: 实现核心分析逻辑
    # from src.analyst import VideoAnalyst
    # analyst = VideoAnalyst(...)
    # result = analyst.analyze(video_path, args.query)
    
    print("\n⏳ 正在分析视频...")
    print("(核心分析逻辑开发中，敬请期待！)")
    
    # 示例输出结构
    result = {
        "video": str(video_path),
        "query": args.query,
        "model": args.model,
        "analyzed_at": datetime.now().isoformat(),
        "segments": [
            {
                "start_time": "00:00:00",
                "end_time": "00:01:00",
                "description": "示例片段 1",
                "relevance": 0.85
            }
        ],
        "summary": "分析完成，发现 X 个相关片段"
    }
    
    # 输出结果
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ 报告已保存：{output_path}")
    else:
        print("\n📊 分析结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\n✨ 分析完成！")


if __name__ == "__main__":
    main()
