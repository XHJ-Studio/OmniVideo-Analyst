#!/usr/bin/env python3
"""
推理引擎模块单元测试

运行测试:
    python -m pytest tests/test_inference.py -v
"""

import pytest
from src.inference import VideoInferenceEngine, SegmentAnalysisResult
from src.model_loader import ModelLoader


class TestSegmentAnalysisResult:
    """SegmentAnalysisResult 测试"""
    
    def test_creation(self):
        """测试结果创建"""
        result = SegmentAnalysisResult(
            segment_index=0,
            start_time=0.0,
            end_time=60.0,
            query="测试查询",
            has_relevant_content=True,
            description="测试描述",
            confidence=0.85,
            raw_response="原始响应"
        )
        
        assert result.segment_index == 0
        assert result.has_relevant_content is True
        assert result.confidence == 0.85
    
    def test_to_dict(self):
        """测试转换为字典"""
        result = SegmentAnalysisResult(
            segment_index=1,
            start_time=60.0,
            end_time=120.0,
            query="查询",
            has_relevant_content=False,
            description="无相关内容",
            confidence=0.3,
            raw_response="响应"
        )
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert data["segment_index"] == 1
        assert data["start_time"] == 60.0
        assert data["has_relevant_content"] is False


class TestVideoInferenceEngine:
    """推理引擎测试"""
    
    @pytest.fixture
    def model_loader(self):
        """创建模型加载器 mock"""
        loader = ModelLoader()
        # 不实际加载模型
        loader.model = None
        loader.processor = None
        return loader
    
    @pytest.fixture
    def engine(self, model_loader):
        """创建推理引擎"""
        return VideoInferenceEngine(model_loader)
    
    def test_init(self, engine):
        """测试初始化"""
        assert engine.model_loader is not None
        assert engine.default_prompt_template == "general"
    
    def test_parse_response_json(self, engine):
        """测试解析 JSON 格式响应"""
        response = """
        {
            "has_relevant": true,
            "description": "在 00:30 处发现红色自行车",
            "confidence": 0.9
        }
        """
        
        parsed = engine._parse_response(response, "找红色自行车")
        
        assert parsed["has_relevant"] is True
        assert "红色自行车" in parsed["description"]
        assert parsed["confidence"] == 0.9
    
    def test_parse_response_text(self, engine):
        """测试解析纯文本响应"""
        response = "在视频中看到一辆红色自行车出现在 30 秒处"
        
        parsed = engine._parse_response(response, "找红色自行车")
        
        assert parsed["has_relevant"] is True
        assert parsed["confidence"] == 0.7
    
    def test_check_relevance_negative(self, engine):
        """测试相关性检测（否定情况）"""
        response = "没有发现，未看到任何相关内容，不存在"
        
        is_relevant = engine._check_relevance(response, "找东西")
        
        assert is_relevant is False
    
    def test_check_relevance_positive(self, engine):
        """测试相关性检测（肯定情况）"""
        response = "在视频中看到一个人打开了门，然后走进了房间"
        
        is_relevant = engine._check_relevance(response, "门有没有打开")
        
        assert is_relevant is True
    
    def test_check_relevance_short(self, engine):
        """测试相关性检测（过短响应）"""
        response = "没有"
        
        is_relevant = engine._check_relevance(response, "找东西")
        
        assert is_relevant is False
    
    def test_parse_segment_path(self, engine):
        """测试从路径提取片段信息"""
        path = "/tmp/video_seg001_0000-0060.mp4"
        
        info = engine._parse_segment_path(path)
        
        assert info["index"] == 1
        assert info["start"] == 0.0
        assert info["end"] == 60.0
    
    def test_parse_segment_path_no_match(self, engine):
        """测试路径解析（无匹配模式）"""
        path = "/tmp/video.mp4"
        
        info = engine._parse_segment_path(path)
        
        assert info["index"] == 0
        assert info["start"] == 0.0
        assert info["end"] == 0.0
    
    def test_build_multimodal_message(self, engine):
        """测试构建多模态消息"""
        messages = engine._build_multimodal_message(
            video_path="/tmp/video.mp4",
            audio_path="/tmp/audio.wav",
            query="测试查询"
        )
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) == 3
        
        content_types = [item["type"] for item in messages[0]["content"]]
        assert "video" in content_types
        assert "audio" in content_types
        assert "text" in content_types


class TestInferenceIntegration:
    """推理集成测试（需要实际模型，跳过）"""
    
    @pytest.mark.skip(reason="需要实际加载模型")
    def test_analyze_segment(self):
        """测试单段分析"""
        pytest.skip("需要 GPU 和已下载的模型")
    
    @pytest.mark.skip(reason="需要实际加载模型")
    def test_batch_analyze(self):
        """测试批量分析"""
        pytest.skip("需要 GPU 和已下载的模型")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
