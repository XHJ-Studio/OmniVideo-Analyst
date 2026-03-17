#!/usr/bin/env python3
"""
视频预处理模块单元测试

运行测试:
    python -m pytest tests/test_preprocessor.py -v
"""

import os
import pytest
from pathlib import Path
from src.preprocessor import VideoPreprocessor, VideoInfo, VideoSegment


class TestVideoPreprocessor:
    """视频预处理器测试"""
    
    @pytest.fixture
    def preprocessor(self, tmp_path):
        """创建预处理器实例（使用临时目录）"""
        return VideoPreprocessor(output_dir=str(tmp_path))
    
    def test_init(self, preprocessor):
        """测试初始化"""
        assert preprocessor.output_dir.exists()
        assert preprocessor.output_dir.is_dir()
    
    def test_get_video_info_not_found(self, preprocessor):
        """测试文件不存在的情况"""
        with pytest.raises(FileNotFoundError):
            preprocessor.get_video_info("/nonexistent/video.mp4")
    
    def test_segment_video_default(self, preprocessor):
        """测试默认切段（需要实际视频文件）"""
        # TODO: 添加实际视频文件测试
        pytest.skip("需要实际视频文件")
    
    def test_extract_keyframes_default(self, preprocessor):
        """测试关键帧提取（需要实际视频文件）"""
        # TODO: 添加实际视频文件测试
        pytest.skip("需要实际视频文件")
    
    def test_extract_audio_default(self, preprocessor):
        """测试音频提取（需要实际视频文件）"""
        # TODO: 添加实际视频文件测试
        pytest.skip("需要实际视频文件")
    
    def test_cleanup(self, preprocessor, tmp_path):
        """测试清理功能"""
        # 创建测试文件
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        assert test_file.exists()
        preprocessor.cleanup(remove_all=True)
        assert not tmp_path.exists()


class TestVideoInfo:
    """VideoInfo 数据类测试"""
    
    def test_video_info_str(self):
        """测试字符串表示"""
        info = VideoInfo(
            path="/test/video.mp4",
            width=1920,
            height=1080,
            fps=30.0,
            duration=120.0,
            frame_count=3600,
            codec="H264",
            has_audio=True
        )
        
        assert "video.mp4" in str(info)
        assert "1920x1080" in str(info)
        assert "30.0fps" in str(info)


class TestVideoSegment:
    """VideoSegment 数据类测试"""
    
    def test_video_segment_creation(self):
        """测试片段创建"""
        segment = VideoSegment(
            index=0,
            start_time=0.0,
            end_time=60.0,
            path="/test/segment.mp4",
            frame_count=1800
        )
        
        assert segment.index == 0
        assert segment.start_time == 0.0
        assert segment.end_time == 60.0
        assert segment.frame_count == 1800


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
