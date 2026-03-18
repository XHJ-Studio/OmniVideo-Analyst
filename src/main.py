#!/usr/bin/env python3
"""
OmniVideo-Analyst CLI 入口（完整版）

整合所有模块：
- preprocessor: 视频预处理（切段、关键帧、音频）
- model_loader: 模型加载
- inference: 视频推理
- aggregator: 结果聚合

使用示例:
    python src/main.py --video surveillance.mp4 --query "有没有人翻越围墙？"
    python src/main.py -v video.mp4 -q "找红色自行车" -o report.json --segment-length 30
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from tqdm import tqdm

from .preprocessor import VideoPreprocessor
from .model_loader import ModelLoader
from .inference import VideoInferenceEngine, SegmentAnalysisResult
from .aggregator import ResultAggregator, AnalysisReport


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="🎬 OmniVideo-Analyst - 使用自然语言分析视频内容",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --video surveillance.mp4 --query "有没有人翻越围墙？"
  %(prog)s -v video.mp4 -q "找红色自行车" -o report.json
  %(prog)s -v meeting.mp4 -q "找出所有提到预算的片段" --segment-length 30
  %(prog)s -v video.mp4 -q "门有没有打开过" --model Qwen/Qwen3-Omni-30B-A3B-Thinking --tp 3
        """
    )
    
    # 必需参数
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
    
    # 输出选项
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出报告文件路径 (JSON/Markdown)，默认输出到控制台"
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "markdown"],
        default="json",
        help="输出格式 (json/markdown)，默认 json"
    )
    
    # 模型选项
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Thinking",
        help="模型名称或本地路径 (默认：Qwen/Qwen3-Omni-30B-A3B-Thinking)"
    )
    
    parser.add_argument(
        "--tensor-parallel-size", "-tp",
        type=int,
        default=1,
        help="张量并行 GPU 数量，默认 1"
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "transformers"],
        default="vllm",
        help="推理后端 (vllm/transformers)，默认 vllm"
    )
    
    # 预处理选项
    parser.add_argument(
        "--segment-length",
        type=int,
        default=60,
        help="视频切段长度 (秒)，默认 60 秒"
    )
    
    parser.add_argument(
        "--extract-keyframes",
        action="store_true",
        help="提取关键帧（用于场景变化检测）"
    )
    
    parser.add_argument(
        "--extract-audio",
        action="store_true",
        help="提取音频轨道（用于音视频联合分析）"
    )
    
    # 分析选项
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="置信度阈值 (0-1)，低于此值的结果会被过滤，默认 0.5"
    )
    
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="general",
        help="Prompt 模板名称 (general/door_window/person/vehicle/suspicious/object)"
    )
    
    # 其他选项
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="空运行模式（不实际加载模型，用于测试流程）"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细输出（DEBUG 日志）"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    return parser.parse_args()


def print_banner():
    """打印欢迎横幅"""
    print("=" * 60)
    print("🎬 OmniVideo-Analyst v0.1.0")
    print("   基于 Qwen3-Omni 的视频智能分析工具")
    print("=" * 60)


def main():
    args = parse_args()
    logger = setup_logging(args.verbose)
    
    print_banner()
    
    # 验证输入文件
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"❌ 视频文件不存在：{video_path}")
        sys.exit(1)
    
    logger.info(f"📹 视频：{video_path.name}")
    logger.info(f"🔍 查询：{args.query}")
    logger.info(f"🧠 模型：{args.model}")
    logger.info(f"📏 切段：{args.segment_length}秒/段")
    logger.info(f"🎮 GPU 并行：{args.tensor_parallel_size}")
    logger.info(f"🎚️ 置信度阈值：{args.confidence_threshold}")
    print("=" * 60)
    
    try:
        # ========== Phase 1: 视频预处理 ==========
        logger.info("【1/4】视频预处理...")
        
        preprocessor = VideoPreprocessor()
        
        # 获取视频信息
        video_info = preprocessor.get_video_info(str(video_path))
        logger.info(f"视频信息：{video_info.width}x{video_info.height}, "
                   f"{video_info.fps}fps, 时长{video_info.duration:.1f}s")
        
        # 切分视频
        logger.info(f"开始切分视频...")
        segments = preprocessor.segment_video(
            str(video_path),
            segment_length=args.segment_length
        )
        logger.info(f"✅ 切分完成：{len(segments)}个片段")
        
        # 提取关键帧（可选）
        keyframes = []
        if args.extract_keyframes:
            logger.info("提取关键帧...")
            keyframes = preprocessor.extract_keyframes(str(video_path))
            logger.info(f"✅ 提取{len(keyframes)}个关键帧")
        
        # 提取音频（可选）
        audio_path = None
        if args.extract_audio:
            logger.info("提取音频...")
            audio_path = preprocessor.extract_audio(str(video_path))
            logger.info(f"✅ 音频路径：{audio_path}")
        
        # ========== Phase 2: 模型加载 ==========
        logger.info("【2/4】加载模型...")
        
        if args.dry_run:
            logger.info("⚠️  空运行模式：跳过模型加载")
            model_loader = ModelLoader()
            engine = VideoInferenceEngine(model_loader)
        else:
            model_loader = ModelLoader()
            
            # 加载模型
            model, processor = model_loader.load_model(
                args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                backend=args.backend
            )
            logger.info(f"✅ 模型加载完成")
            
            # 创建推理引擎
            engine = VideoInferenceEngine(
                model_loader,
                default_prompt_template=args.prompt_template
            )
        
        # ========== Phase 3: 视频推理 ==========
        logger.info("【3/4】分析视频片段...")
        
        results = []
        total_segments = len(segments)
        
        with tqdm(total=total_segments, desc="分析进度") as pbar:
            for i, segment in enumerate(segments):
                try:
                    if args.dry_run:
                        # 空运行模式：生成模拟结果
                        result = SegmentAnalysisResult(
                            segment_index=segment.index,
                            start_time=segment.start_time,
                            end_time=segment.end_time,
                            query=args.query,
                            has_relevant_content=(i % 3 == 0),  # 模拟 1/3 相关
                            description=f"[模拟] 片段{i}的分析结果",
                            confidence=0.5 + (i % 5) * 0.1,
                            raw_response="[模拟响应]"
                        )
                        time.sleep(0.1)  # 模拟延迟
                    else:
                        # 实际分析
                        result = engine.analyze_segment(
                            segment_path=segment.path,
                            query=args.query,
                            prompt_template=args.prompt_template
                        )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"片段{i}分析失败：{e}")
                    # 创建错误结果
                    result = SegmentAnalysisResult(
                        segment_index=segment.index,
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        query=args.query,
                        has_relevant_content=False,
                        description=f"分析失败：{str(e)}",
                        confidence=0.0,
                        raw_response="",
                        metadata={"error": str(e)}
                    )
                    results.append(result)
                
                pbar.update(1)
        
        logger.info(f"✅ 分析完成：{len(results)}个结果")
        
        # ========== Phase 4: 结果聚合 ==========
        logger.info("【4/4】聚合结果...")
        
        aggregator = ResultAggregator(
            confidence_threshold=args.confidence_threshold
        )
        
        report = aggregator.aggregate(
            results,
            video_path=str(video_path),
            video_duration=video_info.duration
        )
        
        logger.info(f"✅ 聚合完成")
        logger.info(f"📊 相关片段：{report.relevant_segments}/{report.analyzed_segments} "
                   f"({report.relevant_segments/report.analyzed_segments*100:.1f}%)")
        logger.info(f"📈 平均置信度：{report.average_confidence:.2f}")
        
        # ========== 输出结果 ==========
        print("\n" + "=" * 60)
        print("📊 分析结果")
        print("=" * 60)
        
        print(f"\n📹 视频：{video_path.name}")
        print(f"🔍 查询：{args.query}")
        print(f"⏱️  时长：{report.video_duration_fmt}")
        print(f"📋 总片段：{report.total_segments}")
        print(f"✅ 相关片段：{report.relevant_segments}/{report.analyzed_segments} "
              f"({report.relevant_segments/report.analyzed_segments*100:.1f}%)")
        print(f"📈 平均置信度：{report.average_confidence:.2f}")
        
        print(f"\n📝 分析总结:")
        print(f"  {report.summary}")
        
        # 显示时间轴
        if report.events:
            print(f"\n🕐 时间轴:")
            print(f"  {'时间':<20} {'描述':<40} {'置信度'}")
            print(f"  {'-'*18} {'-'*38} {'-'*10}")
            for event in report.events[:10]:  # 最多显示 10 个
                time_range = f"{event.start_time_fmt} - {event.end_time_fmt}"
                desc = event.description[:38] + "..." if len(event.description) > 40 else event.description
                print(f"  {time_range:<20} {desc:<40} {event.confidence:.2f}")
            
            if len(report.events) > 10:
                print(f"  ... 还有{len(report.events) - 10}个事件")
        
        # 保存报告
        if args.output:
            output_path = Path(args.output)
            aggregator.save_report(
                report,
                str(output_path),
                format=args.output_format
            )
            print(f"\n✅ 报告已保存：{output_path}")
        
        print("\n" + "=" * 60)
        print("✨ 分析完成！")
        print("=" * 60)
        
        # 清理临时文件（可选）
        # preprocessor.cleanup(remove_all=False)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  用户中断")
        return 130
        
    except Exception as e:
        logger.exception(f"❌ 分析失败：{e}")
        print("\n❌ 分析过程中发生错误，请检查日志")
        return 1


if __name__ == "__main__":
    sys.exit(main())
