#!/usr/bin/env python3
"""
基础视频分析示例

用法:
    python examples/basic_analysis.py --video path/to/video.mp4 --query "有没有人翻越围墙？"
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import VideoPreprocessor
from src.model_loader import ModelLoader
from src.inference import VideoInferenceEngine
from src.aggregator import ResultAggregator


def main():
    parser = argparse.ArgumentParser(description="基础视频分析示例")
    parser.add_argument("--video", "-v", type=str, required=True, help="视频文件路径")
    parser.add_argument("--query", "-q", type=str, required=True, help="查询内容")
    parser.add_argument("--output", "-o", type=str, default="report.json", help="输出报告路径")
    parser.add_argument("--segment-length", type=int, default=60, help="切段长度（秒）")
    parser.add_argument("--dry-run", action="store_true", help="空运行模式")
    
    args = parser.parse_args()
    
    print(f"🎬 OmniVideo-Analyst 基础分析示例")
    print(f"=" * 50)
    print(f"📹 视频：{args.video}")
    print(f"🔍 查询：{args.query}")
    print(f"📄 输出：{args.output}")
    print(f"=" * 50)
    
    # Phase 1: 预处理
    print("\n【1/4】视频预处理...")
    preprocessor = VideoPreprocessor()
    video_info = preprocessor.get_video_info(args.video)
    print(f"  视频信息：{video_info.width}x{video_info.height}, {video_info.duration:.1f}s")
    
    segments = preprocessor.segment_video(args.video, segment_length=args.segment_length)
    print(f"  切分完成：{len(segments)}个片段")
    
    # Phase 2: 模型加载
    print("\n【2/4】加载模型...")
    if args.dry_run:
        print("  ⚠️  空运行模式：跳过模型加载")
        model_loader = ModelLoader()
    else:
        model_loader = ModelLoader()
        model, processor = model_loader.load_model("Qwen/Qwen3-Omni-30B-A3B-Thinking")
        print("  ✅ 模型加载完成")
    
    # Phase 3: 视频推理
    print("\n【3/4】分析视频...")
    engine = VideoInferenceEngine(model_loader)
    results = []
    
    for i, segment in enumerate(segments):
        if args.dry_run:
            # 模拟结果
            from src.inference import SegmentAnalysisResult
            result = SegmentAnalysisResult(
                segment_index=i,
                start_time=segment.start_time,
                end_time=segment.end_time,
                query=args.query,
                has_relevant_content=(i % 3 == 0),
                description=f"[模拟] 片段{i}的结果",
                confidence=0.5,
                raw_response="[模拟]"
            )
        else:
            result = engine.analyze_segment(segment.path, args.query)
        results.append(result)
    
    print(f"  ✅ 分析完成：{len(results)}个结果")
    
    # Phase 4: 结果聚合
    print("\n【4/4】聚合结果...")
    aggregator = ResultAggregator(confidence_threshold=0.5)
    report = aggregator.aggregate(results, video_path=args.video, video_duration=video_info.duration)
    
    print(f"  ✅ 聚合完成")
    print(f"  📊 相关片段：{report.relevant_segments}/{report.analyzed_segments}")
    
    # 保存报告
    aggregator.save_report(report, args.output, format="json")
    print(f"\n✅ 报告已保存：{args.output}")
    
    # 显示总结
    print(f"\n📝 分析总结:")
    print(f"  {report.summary}")


if __name__ == "__main__":
    main()
