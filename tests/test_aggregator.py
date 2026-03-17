#!/usr/bin/env python3
"""
结果聚合模块单元测试

运行测试:
    python -m pytest tests/test_aggregator.py -v
"""

import pytest
from src.aggregator import ResultAggregator, AnalysisReport, TimelineEvent
from src.inference import SegmentAnalysisResult


class TestTimelineEvent:
    """TimelineEvent 测试"""
    
    def test_creation(self):
        """测试事件创建"""
        event = TimelineEvent(
            start_time=60.0,
            end_time=120.0,
            description="测试事件",
            confidence=0.85,
            segment_index=1
        )
        
        assert event.start_time == 60.0
        assert event.end_time == 120.0
        assert event.confidence == 0.85
    
    def test_to_dict(self):
        """测试转换为字典"""
        event = TimelineEvent(
            start_time=60.0,
            end_time=120.0,
            description="测试",
            confidence=0.9,
            segment_index=0
        )
        
        data = event.to_dict()
        
        assert data["start_time"] == 60.0
        assert data["end_time"] == 120.0
        assert data["start_time_fmt"] == "0:01:00"
        assert data["end_time_fmt"] == "0:02:00"
        assert data["confidence"] == 0.9
    
    def test_format_time(self):
        """测试时间格式化"""
        assert TimelineEvent._format_time(0) == "0:00:00"
        assert TimelineEvent._format_time(60) == "0:01:00"
        assert TimelineEvent._format_time(3661) == "1:01:01"


class TestAnalysisReport:
    """AnalysisReport 测试"""
    
    def test_creation(self):
        """测试报告创建"""
        report = AnalysisReport(
            query="测试查询",
            video_path="/test/video.mp4",
            video_duration=900.0,
            total_segments=15,
            analyzed_segments=15,
            relevant_segments=5
        )
        
        assert report.query == "测试查询"
        assert report.total_segments == 15
        assert report.relevant_segments == 5
    
    def test_to_dict(self):
        """测试转换为字典"""
        report = AnalysisReport(
            query="查询",
            video_path="/test.mp4",
            video_duration=120.0,
            total_segments=2,
            analyzed_segments=2,
            relevant_segments=1,
            average_confidence=0.8
        )
        
        data = report.to_dict()
        
        assert data["query"] == "查询"
        assert data["video_duration_fmt"] == "0:02:00"
        assert data["relevance_rate"] == 0.5
        assert data["statistics"]["average_confidence"] == 0.8


class TestResultAggregator:
    """ResultAggregator 测试"""
    
    @pytest.fixture
    def aggregator(self):
        """创建聚合器"""
        return ResultAggregator(confidence_threshold=0.5)
    
    @pytest.fixture
    def sample_results(self):
        """创建示例结果"""
        return [
            SegmentAnalysisResult(
                segment_index=0,
                start_time=0.0,
                end_time=60.0,
                query="测试",
                has_relevant_content=True,
                description="片段 1 描述",
                confidence=0.9,
                raw_response="响应 1"
            ),
            SegmentAnalysisResult(
                segment_index=1,
                start_time=60.0,
                end_time=120.0,
                query="测试",
                has_relevant_content=False,
                description="片段 2 描述",
                confidence=0.3,
                raw_response="响应 2"
            ),
            SegmentAnalysisResult(
                segment_index=2,
                start_time=120.0,
                end_time=180.0,
                query="测试",
                has_relevant_content=True,
                description="片段 3 描述",
                confidence=0.7,
                raw_response="响应 3"
            )
        ]
    
    def test_init(self, aggregator):
        """测试初始化"""
        assert aggregator.confidence_threshold == 0.5
        assert aggregator.min_relevant_segments == 1
    
    def test_aggregate(self, aggregator, sample_results):
        """测试聚合"""
        report = aggregator.aggregate(
            sample_results,
            video_path="/test.mp4",
            video_duration=180.0
        )
        
        assert report.total_segments == 3
        assert report.analyzed_segments == 3
        assert report.relevant_segments == 2  # 2 个相关片段
        assert len(report.events) == 2
    
    def test_aggregate_empty(self, aggregator):
        """测试空结果聚合"""
        report = aggregator.aggregate(
            [],
            video_path="/test.mp4",
            video_duration=60.0
        )
        
        assert report.total_segments == 0
        assert report.relevant_segments == 0
        assert "未进行分析" in report.summary
    
    def test_filter_by_confidence(self, aggregator, sample_results):
        """测试置信度过滤"""
        filtered = aggregator._filter_by_confidence(sample_results)
        
        # 置信度 0.3 的应该被过滤掉
        assert len(filtered) == 2
        assert all(r.confidence >= 0.5 for r in filtered)
    
    def test_extract_events(self, aggregator, sample_results):
        """测试事件提取"""
        events = aggregator._extract_events(sample_results)
        
        # 只提取 has_relevant_content=True 的
        assert len(events) == 2
        assert all(e.confidence > 0 for e in events)
    
    def test_generate_summary_positive(self, aggregator, sample_results):
        """测试生成总结（正面情况）"""
        report = aggregator.aggregate(sample_results, video_duration=180.0)
        summary = aggregator._generate_summary(report)
        
        assert "相关片段" in summary
        assert "置信度" in summary
    
    def test_generate_summary_empty(self, aggregator):
        """测试生成总结（空结果）"""
        report = aggregator.aggregate([], video_duration=60.0)
        
        assert "未进行" in report.summary or "未发现" in report.summary
    
    def test_save_report_json(self, aggregator, sample_results, tmp_path):
        """测试保存 JSON 报告"""
        report = aggregator.aggregate(sample_results, video_duration=180.0)
        output_path = tmp_path / "report.json"
        
        aggregator.save_report(report, str(output_path), format="json")
        
        assert output_path.exists()
        
        import json
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert data["query"] == "测试"
        assert data["relevant_segments"] == 2
    
    def test_save_report_markdown(self, aggregator, sample_results, tmp_path):
        """测试保存 Markdown 报告"""
        report = aggregator.aggregate(sample_results, video_duration=180.0)
        output_path = tmp_path / "report.md"
        
        aggregator.save_report(report, str(output_path), format="markdown")
        
        assert output_path.exists()
        
        content = output_path.read_text(encoding='utf-8')
        assert "# 视频分析报告" in content
        assert "## 📊 统计信息" in content
        assert "## 🕐 时间轴" in content


class TestResultAggregatorThreshold:
    """不同阈值测试"""
    
    def test_high_threshold(self):
        """高阈值测试"""
        aggregator = ResultAggregator(confidence_threshold=0.8)
        
        results = [
            SegmentAnalysisResult(
                segment_index=0,
                start_time=0.0,
                end_time=60.0,
                query="测试",
                has_relevant_content=True,
                description="高置信度",
                confidence=0.9,
                raw_response=""
            ),
            SegmentAnalysisResult(
                segment_index=1,
                start_time=60.0,
                end_time=120.0,
                query="测试",
                has_relevant_content=True,
                description="低置信度",
                confidence=0.5,
                raw_response=""
            )
        ]
        
        filtered = aggregator._filter_by_confidence(results)
        
        # 只有 0.9 的通过
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.9
    
    def test_low_threshold(self):
        """低阈值测试"""
        aggregator = ResultAggregator(confidence_threshold=0.3)
        
        results = [
            SegmentAnalysisResult(
                segment_index=0,
                start_time=0.0,
                end_time=60.0,
                query="测试",
                has_relevant_content=True,
                description="测试",
                confidence=0.5,
                raw_response=""
            )
        ]
        
        filtered = aggregator._filter_by_confidence(results)
        
        # 0.5 > 0.3，应该通过
        assert len(filtered) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
