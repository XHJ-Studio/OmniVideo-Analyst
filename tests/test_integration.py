#!/usr/bin/env python3
"""
整合测试

测试整个分析流程的端到端功能
"""

import pytest
from pathlib import Path


class TestIntegration:
    """整合测试"""
    
    @pytest.mark.skip(reason="需要实际视频文件和 GPU")
    def test_full_analysis_pipeline(self):
        """测试完整分析流程"""
        from src.preprocessor import VideoPreprocessor
        from src.model_loader import ModelLoader
        from src.inference import VideoInferenceEngine
        from src.aggregator import ResultAggregator
        
        # 1. 预处理
        preprocessor = VideoPreprocessor()
        # 需要实际视频文件
        pytest.skip("需要实际视频文件")
        
        # 2. 模型加载
        # 3. 推理
        # 4. 聚合
    
    @pytest.mark.skip(reason="需要实际视频文件")
    def test_cli_dry_run(self):
        """测试 CLI 空运行模式"""
        import subprocess
        
        result = subprocess.run(
            ["python", "src/main.py", "--video", "test.mp4", "--query", "test", "--dry-run"],
            capture_output=True,
            text=True
        )
        
        # 应该成功完成（虽然是空运行）
        assert result.returncode == 0
    
    @pytest.mark.skip(reason="需要运行 API 服务")
    def test_api_health(self):
        """测试 API 健康检查"""
        import requests
        
        response = requests.get("http://localhost:8000/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
