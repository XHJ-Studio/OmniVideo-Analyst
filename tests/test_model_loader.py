#!/usr/bin/env python3
"""
模型加载模块单元测试

运行测试:
    python -m pytest tests/test_model_loader.py -v
"""

import os
import pytest
from pathlib import Path
from src.model_loader import ModelLoader, ModelConfig


class TestModelLoader:
    """模型加载器测试"""
    
    @pytest.fixture
    def loader(self, tmp_path):
        """创建模型加载器实例（使用临时缓存目录）"""
        return ModelLoader(cache_dir=str(tmp_path))
    
    def test_init(self, loader):
        """测试初始化"""
        assert loader.cache_dir.exists()
        assert loader.cache_dir.is_dir()
    
    def test_get_model_info(self, loader):
        """测试获取模型信息"""
        info = loader.get_model_info("Qwen/Qwen3-Omni-30B-A3B-Thinking")
        
        assert info["supported"] is True
        assert info["type"] == "thinking"
        assert info["has_video"] is True
    
    def test_get_model_info_unsupported(self, loader):
        """测试获取不支持的模型信息"""
        info = loader.get_model_info("Unknown/Model")
        
        assert info["supported"] is False
    
    def test_list_available_models(self, loader):
        """测试列出可用模型"""
        models = loader.list_available_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        assert "Qwen/Qwen3-Omni-30B-A3B-Thinking" in models
    
    def test_download_model_not_found(self, loader):
        """测试下载模型（需要网络，跳过实际下载）"""
        pytest.skip("需要网络连接")
    
    def test_load_model_vllm(self, loader):
        """测试使用 vLLM 加载模型（需要 GPU，跳过实际加载）"""
        pytest.skip("需要 GPU 和已下载的模型")
    
    def test_load_model_transformers(self, loader):
        """测试使用 Transformers 加载模型（需要 GPU，跳过实际加载）"""
        pytest.skip("需要 GPU 和已下载的模型")
    
    def test_unload_model(self, loader):
        """测试卸载模型"""
        # 卸载空模型应该不会报错
        loader.unload_model()
        assert loader.model is None
        assert loader.processor is None
    
    def test_model_config_str(self):
        """测试 ModelConfig 字符串表示"""
        config = ModelConfig(
            model_name="Qwen/Test",
            model_path=None,
            tensor_parallel_size=3,
            dtype="auto",
            max_model_len=32768,
            gpu_memory_utilization=0.9,
            trust_remote_code=True
        )
        
        config_str = str(config)
        assert "Qwen/Test" in config_str
        assert "tp=3" in config_str


class TestSupportedModels:
    """支持模型列表测试"""
    
    def test_thinking_model(self):
        """测试 Thinking 模型配置"""
        from src.model_loader import ModelLoader
        
        loader = ModelLoader()
        info = loader.get_model_info("Qwen/Qwen3-Omni-30B-A3B-Thinking")
        
        assert info["type"] == "thinking"
        assert info["has_video"] is True
        assert info["default_tp"] == 3
    
    def test_instruct_model(self):
        """测试 Instruct 模型配置"""
        from src.model_loader import ModelLoader
        
        loader = ModelLoader()
        info = loader.get_model_info("Qwen/Qwen3-Omni-30B-A3B-Instruct")
        
        assert info["type"] == "instruct"
        assert info["has_audio"] is True
        assert info["has_video"] is True
    
    def test_quantized_model(self):
        """测试量化模型配置"""
        from src.model_loader import ModelLoader
        
        loader = ModelLoader()
        info = loader.get_model_info("cpatonn-mirror/Qwen3-Omni-30B-A3B-Thinking-AWQ-8bit")
        
        assert info["quantization"] == "awq"
        assert info["bits"] == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
