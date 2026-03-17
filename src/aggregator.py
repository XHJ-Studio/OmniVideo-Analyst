#!/usr/bin/env python3
"""
结果聚合模块

功能:
- 多段结果合并
- 时间轴生成
- 置信度计算与过滤
- JSON/Markdown 报告输出

使用示例:
    from src.aggregator import ResultAggregator
    
    aggregator = ResultAggregator()
    report = aggregator.aggregate(results, video_duration=900)
    aggregator.save_report(report, "output.json")
    aggregator.export_timeline(report, "timeline.md")
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .inference import SegmentAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class TimelineEvent:
    """时间轴事件"""
    start_time: float
    end_time: float
    description: str
    confidence: float
    segment_index: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_time_fmt": self._format_time(self.start_time),
            "end_time_fmt": self._format_time(self.end_time),
            "description": self.description,
            "confidence": self.confidence,
            "segment_index": self.segment_index
        }
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间为 HH:MM:SS"""
        return str(timedelta(seconds=int(seconds)))


@dataclass
class AnalysisReport:
    """分析报告"""
    query: str
    video_path: str
    video_duration: float
    total_segments: int
    analyzed_segments: int
    relevant_segments: int
    
    # 时间轴事件列表
    events: List[TimelineEvent] = field(default_factory=list)
    
    # 统计信息
    average_confidence: float = 0.0
    max_confidence: float = 0.0
    min_confidence: float = 0.0
    
    # 总结
    summary: str = ""
    
    # 原始结果
    raw_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "query": self.query,
            "video_path": self.video_path,
            "video_duration": self.video_duration,
            "video_duration_fmt": str(timedelta(seconds=int(self.video_duration))),
            "total_segments": self.total_segments,
            "analyzed_segments": self.analyzed_segments,
            "relevant_segments": self.relevant_segments,
            "relevance_rate": self.relevant_segments / self.analyzed_segments if self.analyzed_segments > 0 else 0,
            "events": [e.to_dict() for e in self.events],
            "statistics": {
                "average_confidence": self.average_confidence,
                "max_confidence": self.max_confidence,
                "min_confidence": self.min_confidence
            },
            "summary": self.summary,
            "raw_results": self.raw_results
        }


class ResultAggregator:
    """结果聚合器"""
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        min_relevant_segments: int = 1
    ):
        """
        初始化聚合器
        
        Args:
            confidence_threshold: 置信度阈值，低于此值的结果会被过滤
            min_relevant_segments: 最小相关片段数
        """
        self.confidence_threshold = confidence_threshold
        self.min_relevant_segments = min_relevant_segments
        
        logger.info(f"聚合器初始化完成，置信度阈值={confidence_threshold}")
    
    def aggregate(
        self,
        results: List[SegmentAnalysisResult],
        video_path: str = "",
        video_duration: float = 0.0
    ) -> AnalysisReport:
        """
        聚合分析结果
        
        Args:
            results: 分析结果列表
            video_path: 视频文件路径
            video_duration: 视频总时长（秒）
            
        Returns:
            AnalysisReport: 聚合报告
        """
        logger.info(f"聚合 {len(results)} 个分析结果")
        
        if not results:
            logger.warning("没有分析结果")
            return self._create_empty_report(video_path, video_duration)
        
        # 过滤低置信度结果
        filtered_results = self._filter_by_confidence(results)
        
        # 提取相关事件
        events = self._extract_events(filtered_results)
        
        # 计算统计信息
        confidences = [r.confidence for r in filtered_results if r.has_relevant_content]
        
        # 创建报告
        report = AnalysisReport(
            query=results[0].query if results else "",
            video_path=video_path,
            video_duration=video_duration,
            total_segments=len(results),
            analyzed_segments=len(filtered_results),
            relevant_segments=len(events),
            events=events,
            average_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            max_confidence=max(confidences) if confidences else 0.0,
            min_confidence=min(confidences) if confidences else 0.0,
            raw_results=[r.to_dict() for r in filtered_results]
        )
        
        # 生成总结
        report.summary = self._generate_summary(report)
        
        logger.info(f"聚合完成：相关片段={report.relevant_segments}/{report.analyzed_segments}")
        return report
    
    def save_report(
        self,
        report: AnalysisReport,
        output_path: str,
        format: str = "json"
    ):
        """
        保存报告
        
        Args:
            report: 分析报告
            output_path: 输出文件路径
            format: 输出格式（json/markdown）
        """
        path = Path(output_path)
        
        if format == "json":
            self._save_json(report, path)
        elif format == "markdown":
            self._save_markdown(report, path)
        else:
            raise ValueError(f"不支持的格式：{format}")
        
        logger.info(f"报告已保存：{output_path}")
    
    def export_timeline(
        self,
        report: AnalysisReport,
        output_path: str,
        include_raw: bool = False
    ):
        """
        导出时间轴（Markdown 格式）
        
        Args:
            report: 分析报告
            output_path: 输出文件路径
            include_raw: 是否包含原始结果
        """
        path = Path(output_path)
        
        lines = []
        lines.append(f"# 视频分析报告\n")
        lines.append(f"**查询**: {report.query}\n")
        lines.append(f"**视频**: {Path(report.video_path).name if report.video_path else '未知'}\n")
        lines.append(f"**时长**: {report.video_duration_fmt}\n")
        lines.append(f"**分析时间**: {timedelta(seconds=int(report.video_duration))}\n")
        lines.append("")
        
        # 统计信息
        lines.append("## 📊 统计信息\n")
        lines.append(f"- 总片段数：{report.total_segments}")
        lines.append(f"- 分析片段数：{report.analyzed_segments}")
        lines.append(f"- 相关片段数：{report.relevant_segments}")
        lines.append(f"- 相关率：{report.relevant_segments/report.analyzed_segments*100:.1f}%")
        lines.append(f"- 平均置信度：{report.average_confidence:.2f}")
        lines.append("")
        
        # 总结
        lines.append("## 📝 分析总结\n")
        lines.append(f"{report.summary}\n")
        lines.append("")
        
        # 时间轴
        if report.events:
            lines.append("## 🕐 时间轴\n")
            lines.append("| 时间 | 描述 | 置信度 |")
            lines.append("|------|------|--------|")
            
            for event in report.events:
                time_range = f"{event.start_time_fmt} - {event.end_time_fmt}"
                desc = event.description[:50] + "..." if len(event.description) > 50 else event.description
                lines.append(f"| {time_range} | {desc} | {event.confidence:.2f} |")
            
            lines.append("")
        
        # 详细事件
        if report.events:
            lines.append("## 📍 详细事件\n")
            
            for i, event in enumerate(report.events, 1):
                lines.append(f"### 事件 {i}")
                lines.append(f"- **时间**: {event.start_time_fmt} - {event.end_time_fmt}")
                lines.append(f"- **置信度**: {event.confidence:.2f}")
                lines.append(f"- **描述**: {event.description}")
                lines.append("")
        
        # 原始结果
        if include_raw and report.raw_results:
            lines.append("## 📄 原始结果\n")
            lines.append("```json")
            lines.append(json.dumps(report.raw_results, ensure_ascii=False, indent=2))
            lines.append("```")
        
        # 写入文件
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"时间轴已导出：{output_path}")
    
    def _create_empty_report(self, video_path: str, video_duration: float) -> AnalysisReport:
        """创建空报告"""
        return AnalysisReport(
            query="",
            video_path=video_path,
            video_duration=video_duration,
            total_segments=0,
            analyzed_segments=0,
            relevant_segments=0,
            summary="未进行分析或没有有效结果"
        )
    
    def _filter_by_confidence(
        self,
        results: List[SegmentAnalysisResult]
    ) -> List[SegmentAnalysisResult]:
        """
        按置信度过滤结果
        
        Args:
            results: 分析结果列表
            
        Returns:
            List[SegmentAnalysisResult]: 过滤后的结果
        """
        filtered = [
            r for r in results
            if r.confidence >= self.confidence_threshold
        ]
        
        logger.info(f"置信度过滤：{len(results)} -> {len(filtered)} (阈值={self.confidence_threshold})")
        return filtered
    
    def _extract_events(
        self,
        results: List[SegmentAnalysisResult]
    ) -> List[TimelineEvent]:
        """
        提取时间轴事件
        
        Args:
            results: 分析结果列表
            
        Returns:
            List[TimelineEvent]: 事件列表
        """
        events = []
        
        for result in results:
            if result.has_relevant_content:
                event = TimelineEvent(
                    start_time=result.start_time,
                    end_time=result.end_time,
                    description=result.description,
                    confidence=result.confidence,
                    segment_index=result.segment_index
                )
                events.append(event)
        
        # 按时间排序
        events.sort(key=lambda e: e.start_time)
        
        logger.info(f"提取 {len(events)} 个时间轴事件")
        return events
    
    def _generate_summary(self, report: AnalysisReport) -> str:
        """
        生成分析总结
        
        Args:
            report: 分析报告
            
        Returns:
            str: 总结文本
        """
        if report.relevant_segments == 0:
            return "在视频中未发现与查询相关的内容。"
        
        # 计算相关率
        relevance_rate = report.relevant_segments / report.analyzed_segments * 100
        
        # 根据相关率生成总结
        if relevance_rate > 50:
            summary = f"视频中**大量出现**与查询相关的内容，共发现 {report.relevant_segments} 个相关片段，"
        elif relevance_rate > 20:
            summary = f"视频中**有部分**与查询相关的内容，共发现 {report.relevant_segments} 个相关片段，"
        else:
            summary = f"视频中**少有**与查询相关的内容，共发现 {report.relevant_segments} 个相关片段，"
        
        summary += f"相关率为 {relevance_rate:.1f}%。"
        
        # 添加置信度信息
        if report.average_confidence > 0.8:
            summary += " 分析结果置信度**很高**。"
        elif report.average_confidence > 0.6:
            summary += " 分析结果置信度**中等**。"
        else:
            summary += " 分析结果置信度**较低**，建议人工复核。"
        
        # 添加第一个事件的简要描述
        if report.events:
            first_event = report.events[0]
            summary += f"\n\n首次出现在 **{first_event.start_time_fmt}**。"
        
        return summary
    
    def _save_json(self, report: AnalysisReport, path: Path):
        """保存为 JSON"""
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    
    def _save_markdown(self, report: AnalysisReport, path: Path):
        """保存为 Markdown"""
        if path.suffix != ".md":
            path = path.with_suffix(".md")
        
        self.export_timeline(report, str(path), include_raw=True)


# 命令行工具
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="结果聚合工具")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入 JSON 文件（分析结果）")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出文件路径")
    parser.add_argument("--format", "-f", type=str, default="json", choices=["json", "markdown"])
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--video-duration", type=float, default=0.0, help="视频总时长（秒）")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 加载输入
    with open(args.input, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    # 转换为 SegmentAnalysisResult
    results = [
        SegmentAnalysisResult(
            segment_index=r["segment_index"],
            start_time=r["start_time"],
            end_time=r["end_time"],
            query=r["query"],
            has_relevant_content=r["has_relevant_content"],
            description=r["description"],
            confidence=r["confidence"],
            raw_response=r.get("raw_response", ""),
            metadata=r.get("metadata", {})
        )
        for r in results_data
    ]
    
    # 创建聚合器
    aggregator = ResultAggregator(confidence_threshold=args.threshold)
    
    # 聚合结果
    report = aggregator.aggregate(
        results,
        video_path="",
        video_duration=args.video_duration
    )
    
    # 保存报告
    aggregator.save_report(report, args.output, format=args.format)
    
    print(f"✅ 报告已保存：{args.output}")
    print(f"📊 相关片段：{report.relevant_segments}/{report.analyzed_segments}")
    print(f"📈 平均置信度：{report.average_confidence:.2f}")
