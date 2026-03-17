#!/usr/bin/env python3
"""
模型加载模块

功能:
- 模型下载（ModelScope / HuggingFace）
- 模型加载（vLLM / Transformers）
- 多 GPU 张量并行配置
- 显存优化（量化、offload）

使用示例:
    from src.model_loader import ModelLoader
    
    loader = ModelLoader()
    model, processor = loader.load_model(
        "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        tensor_parallel_size=3,
        dtype="auto"
    )
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str
    model_path: Optional[str]
    tensor_parallel_size: int
    dtype: str
    max_model_len: int
    gpu_memory_utilization: float
    trust_remote_code: bool
    
    def __str__(self) -> str:
        return (f"ModelConfig({self.model_name}, "
                f"tp={self.tensor_parallel_size}, "
                f"dtype={self.dtype})")


class ModelLoader:
    """模型加载器"""
    
    # 支持的模型列表
    SUPPORTED_MODELS = {
        "Qwen/Qwen3-Omni-30B-A3B-Thinking": {
            "type": "thinking",
            "has_audio": False,
            "has_video": True,
            "default_tp": 3,
        },
        "Qwen/Qwen3-Omni-30B-A3B-Instruct": {
            "type": "instruct",
            "has_audio": True,
            "has_video": True,
            "default_tp": 3,
        },
        "Qwen/Qwen3-Omni-30B-A3B-Captioner": {
            "type": "captioner",
            "has_audio": True,
            "has_video": False,
            "default_tp": 2,
        },
        # 量化版本
        "cpatonn-mirror/Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit": {
            "type": "thinking",
            "quantization": "awq",
            "bits": 4,
            "default_tp": 2,
        },
        "cpatonn-mirror/Qwen3-Omni-30B-A3B-Thinking-AWQ-8bit": {
            "type": "thinking",
            "quantization": "awq",
            "bits": 8,
            "default_tp": 2,
        },
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化模型加载器
        
        Args:
            cache_dir: 模型缓存目录，默认为 ~/.cache/omnivideo
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/omnivideo")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.processor = None
        self.current_config: Optional[ModelConfig] = None
        
        logger.info(f"模型加载器初始化完成，缓存目录：{self.cache_dir}")
    
    def download_model(
        self,
        model_name: str,
        source: str = "modelscope",
        local_dir: Optional[str] = None
    ) -> str:
        """
        下载模型
        
        Args:
            model_name: 模型名称（如 Qwen/Qwen3-Omni-30B-A3B-Thinking）
            source: 下载源（modelscope / huggingface）
            local_dir: 本地保存目录，默认使用缓存目录
            
        Returns:
            str: 模型本地路径
            
        Raises:
            ValueError: 不支持的模型或下载源
        """
        if local_dir is None:
            # 使用模型名作为子目录
            model_safe_name = model_name.replace("/", "_")
            local_dir = str(self.cache_dir / model_safe_name)
        
        logger.info(f"开始下载模型：{model_name}")
        logger.info(f"下载源：{source}")
        logger.info(f"保存目录：{local_dir}")
        
        # 检查是否已下载
        if Path(local_dir).exists():
            logger.info(f"模型已存在：{local_dir}")
            return local_dir
        
        try:
            if source == "modelscope":
                self._download_from_modelscope(model_name, local_dir)
            elif source == "huggingface":
                self._download_from_huggingface(model_name, local_dir)
            else:
                raise ValueError(f"不支持的下载源：{source}")
            
            logger.info(f"模型下载完成：{local_dir}")
            return local_dir
            
        except Exception as e:
            logger.error(f"模型下载失败：{e}")
            raise RuntimeError(f"模型下载失败：{e}")
    
    def load_model(
        self,
        model_name: str,
        tensor_parallel_size: Optional[int] = None,
        dtype: str = "auto",
        backend: str = "vllm",
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        加载模型
        
        Args:
            model_name: 模型名称或本地路径
            tensor_parallel_size: 张量并行 GPU 数量
            dtype: 数据类型（auto/float16/bfloat16）
            backend: 推理后端（vllm / transformers）
            **kwargs: 其他配置参数
            
        Returns:
            Tuple[Any, Any]: (模型实例，Processor 实例)
            
        Raises:
            ValueError: 不支持的后端或模型
        """
        # 检查是否是本地路径
        if Path(model_name).exists():
            model_path = str(model_name)
            # 从路径推断模型名
            model_name = self._infer_model_name(model_path)
        else:
            model_path = None
        
        # 获取模型配置
        model_info = self.SUPPORTED_MODELS.get(model_name, {})
        
        # 自动设置 tensor_parallel_size
        if tensor_parallel_size is None:
            tensor_parallel_size = model_info.get("default_tp", 1)
        
        # 创建配置
        config = ModelConfig(
            model_name=model_name,
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=kwargs.get("max_model_len", 32768),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
            trust_remote_code=kwargs.get("trust_remote_code", True)
        )
        
        logger.info(f"加载模型配置：{config}")
        
        # 根据后端选择加载方式
        if backend == "vllm":
            return self._load_with_vllm(config)
        elif backend == "transformers":
            return self._load_with_transformers(config)
        else:
            raise ValueError(f"不支持的推理后端：{backend}")
    
    def unload_model(self):
        """卸载模型，释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        self.current_config = None
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        # 清理 CUDA 缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("模型已卸载，显存已释放")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            Dict[str, Any]: 模型信息字典
        """
        info = self.SUPPORTED_MODELS.get(model_name, {})
        return {
            "model_name": model_name,
            "supported": model_name in self.SUPPORTED_MODELS,
            **info
        }
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        列出所有可用模型
        
        Returns:
            Dict[str, Dict[str, Any]]: 模型信息字典
        """
        return self.SUPPORTED_MODELS.copy()
    
    def _download_from_modelscope(self, model_name: str, local_dir: str):
        """从 ModelScope 下载模型"""
        try:
            from modelscope import snapshot_download
            
            snapshot_download(
                model_name,
                local_dir=local_dir,
                revision="master"
            )
        except ImportError:
            raise ImportError("请先安装 modelscope: pip install modelscope")
    
    def _download_from_huggingface(self, model_name: str, local_dir: str):
        """从 HuggingFace 下载模型"""
        try:
            from huggingface_hub import snapshot_download
            
            snapshot_download(
                repo_id=model_name,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
        except ImportError:
            raise ImportError("请先安装 huggingface_hub: pip install huggingface_hub")
    
    def _load_with_vllm(self, config: ModelConfig) -> Tuple[Any, Any]:
        """使用 vLLM 加载模型"""
        logger.info("使用 vLLM 后端加载模型...")
        
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "请先安装 vLLM（Qwen3-Omni 分支）:\n"
                "git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git\n"
                "cd vllm && pip install -e ."
            )
        
        # 确定模型路径
        model_path = config.model_path
        if model_path is None:
            # 尝试从缓存加载
            model_safe_name = config.model_name.replace("/", "_")
            cached_path = self.cache_dir / model_safe_name
            if cached_path.exists():
                model_path = str(cached_path)
            else:
                # 自动下载
                logger.info(f"模型未缓存，开始下载：{config.model_name}")
                model_path = self.download_model(config.model_name)
        
        # 创建 vLLM 实例
        logger.info(f"初始化 vLLM 引擎...")
        logger.info(f"  模型路径：{model_path}")
        logger.info(f"  张量并行：{config.tensor_parallel_size}")
        logger.info(f"  数据类型：{config.dtype}")
        
        llm = LLM(
            model=model_path,
            trust_remote_code=config.trust_remote_code,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            seed=42,
        )
        
        # 创建 processor
        processor = self._create_processor(config.model_name, model_path)
        
        self.model = llm
        self.processor = processor
        self.current_config = config
        
        logger.info(f"模型加载完成：{config.model_name}")
        return llm, processor
    
    def _load_with_transformers(self, config: ModelConfig) -> Tuple[Any, Any]:
        """使用 Transformers 加载模型"""
        logger.info("使用 Transformers 后端加载模型...")
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError:
            raise ImportError("请先安装 transformers: pip install transformers")
        
        # 确定模型路径
        model_path = config.model_path
        if model_path is None:
            model_safe_name = config.model_name.replace("/", "_")
            cached_path = self.cache_dir / model_safe_name
            if cached_path.exists():
                model_path = str(cached_path)
            else:
                model_path = self.download_model(config.model_name)
        
        # 设置 dtype
        if config.dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = getattr(torch, config.dtype)
        
        # 加载模型
        logger.info(f"加载模型：{model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=config.trust_remote_code,
            attn_implementation="flash_attention_2"
        )
        
        # 加载 processor
        processor = self._create_processor(config.model_name, model_path)
        
        self.model = model
        self.processor = processor
        self.current_config = config
        
        logger.info(f"模型加载完成：{config.model_name}")
        return model, processor
    
    def _create_processor(self, model_name: str, model_path: str) -> Any:
        """创建 Processor"""
        logger.info(f"创建 Processor：{model_name}")
        
        try:
            # 尝试从 modelscope 导入
            from modelscope import Qwen3OmniMoeProcessor
            return Qwen3OmniMoeProcessor.from_pretrained(model_path)
        except (ImportError, Exception):
            # 回退到 transformers
            from transformers import AutoProcessor
            return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    def _infer_model_name(self, model_path: str) -> str:
        """从本地路径推断模型名称"""
        path = Path(model_path)
        
        # 检查是否是缓存目录
        if path.parent == self.cache_dir:
            # 从目录名还原模型名
            return path.name.replace("_", "/")
        
        # 无法推断，返回路径
        return str(path)


# 命令行工具
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="模型加载工具")
    parser.add_argument("--model", "-m", type=str, required=True, help="模型名称")
    parser.add_argument("--source", type=str, default="modelscope", help="下载源（modelscope/huggingface）")
    parser.add_argument("--tp", type=int, default=None, help="张量并行 GPU 数量")
    parser.add_argument("--backend", type=str, default="vllm", help="推理后端（vllm/transformers）")
    parser.add_argument("--download-only", action="store_true", help="仅下载模型，不加载")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 创建加载器
    loader = ModelLoader()
    
    if args.download_only:
        # 仅下载
        print(f"📥 下载模型：{args.model}")
        local_path = loader.download_model(args.model, source=args.source)
        print(f"✅ 模型已保存到：{local_path}")
    else:
        # 下载并加载
        print(f"🧠 加载模型：{args.model}")
        print(f"🎮 后端：{args.backend}")
        print(f"🔢 张量并行：{args.tp or 'auto'}")
        
        model, processor = loader.load_model(
            args.model,
            tensor_parallel_size=args.tp,
            backend=args.backend
        )
        
        print(f"✅ 模型加载完成！")
        print(f"   模型类型：{type(model)}")
        print(f"   Processor: {type(processor)}")
