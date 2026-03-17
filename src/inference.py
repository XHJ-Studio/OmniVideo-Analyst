#!/usr/bin/env python3
"""
推理引擎模块

功能:
- 单段视频分析
- Prompt 构建与优化
- 多模态输入处理（视频 + 音频 + 文本）
- 结果解析（JSON 提取）

使用示例:
    from src.inference import VideoInferenceEngine
    
    engine = VideoInferenceEngine(model_loader)
    result = engine.analyze_segment(
        segment_path="video_seg001.mp4",
        query="视频里有没有人打开过门？"
    )
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .model_loader import ModelLoader
from .utils.prompt import get_prompt

logger = logging.getLogger(__name__)


@dataclass
class SegmentAnalysisResult:
    """单段分析结果"""
    segment_index: int
    start_time: float
    end_time: float
    query: str
    has_relevant_content: bool
    description: str
    confidence: float  # 0-1
    raw_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "segment_index": self.segment_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "query": self.query,
            "has_relevant_content": self.has_relevant_content,
            "description": self.description,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class VideoInferenceEngine:
    """视频推理引擎"""
    
    def __init__(
        self,
        model_loader: ModelLoader,
        default_prompt_template: str = "general"
    ):
        """
        初始化推理引擎
        
        Args:
            model_loader: 模型加载器实例
            default_prompt_template: 默认 Prompt 模板名称
        """
        self.model_loader = model_loader
        self.default_prompt_template = default_prompt_template
        
        logger.info(f"推理引擎初始化完成，默认模板：{default_prompt_template}")
    
    def analyze_segment(
        self,
        segment_path: str,
        query: str,
        prompt_template: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        **kwargs
    ) -> SegmentAnalysisResult:
        """
        分析单个视频片段
        
        Args:
            segment_path: 视频片段路径
            query: 用户查询
            prompt_template: Prompt 模板名称
            custom_prompt: 自定义 Prompt（覆盖模板）
            **kwargs: 其他参数
            
        Returns:
            SegmentAnalysisResult: 分析结果
        """
        logger.info(f"分析视频片段：{segment_path}")
        logger.info(f"查询：{query}")
        
        # 确定 Prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            template = prompt_template or self.default_prompt_template
            prompt = get_prompt(template, query=query, object_name=query)
        
        # 执行推理
        raw_response = self._run_inference(segment_path, prompt, **kwargs)
        
        # 解析结果
        parsed = self._parse_response(raw_response, query)
        
        # 创建结果对象
        # 从 segment_path 提取时间信息（假设格式：xxx_seg001_0000-0060.mp4）
        segment_info = self._parse_segment_path(segment_path)
        
        result = SegmentAnalysisResult(
            segment_index=segment_info.get("index", 0),
            start_time=segment_info.get("start", 0.0),
            end_time=segment_info.get("end", 0.0),
            query=query,
            has_relevant_content=parsed["has_relevant"],
            description=parsed["description"],
            confidence=parsed["confidence"],
            raw_response=raw_response,
            metadata=parsed.get("metadata", {})
        )
        
        logger.info(f"分析完成：相关={result.has_relevant_content}, 置信度={result.confidence:.2f}")
        return result
    
    def analyze_with_audio(
        self,
        video_path: str,
        audio_path: str,
        query: str,
        **kwargs
    ) -> SegmentAnalysisResult:
        """
        分析带音频的视频片段
        
        Args:
            video_path: 视频文件路径
            audio_path: 音频文件路径
            query: 用户查询
            **kwargs: 其他参数
            
        Returns:
            SegmentAnalysisResult: 分析结果
        """
        logger.info(f"分析音视频片段：视频={video_path}, 音频={audio_path}")
        
        # 构建多模态输入
        messages = self._build_multimodal_message(video_path, audio_path, query)
        
        # 执行推理
        raw_response = self._run_multimodal_inference(messages, **kwargs)
        
        # 解析结果
        parsed = self._parse_response(raw_response, query)
        
        segment_info = self._parse_segment_path(video_path)
        
        result = SegmentAnalysisResult(
            segment_index=segment_info.get("index", 0),
            start_time=segment_info.get("start", 0.0),
            end_time=segment_info.get("end", 0.0),
            query=query,
            has_relevant_content=parsed["has_relevant"],
            description=parsed["description"],
            confidence=parsed["confidence"],
            raw_response=raw_response,
            metadata={
                **parsed.get("metadata", {}),
                "has_audio": True,
                "audio_path": audio_path
            }
        )
        
        logger.info(f"音视频分析完成：相关={result.has_relevant_content}")
        return result
    
    def batch_analyze(
        self,
        segments: List[Dict[str, Any]],
        query: str,
        progress_callback: Optional[callable] = None
    ) -> List[SegmentAnalysisResult]:
        """
        批量分析多个片段
        
        Args:
            segments: 片段列表 [{"path": "...", "start": 0.0, "end": 60.0, "index": 0}, ...]
            query: 用户查询
            progress_callback: 进度回调函数 callback(current, total)
            
        Returns:
            List[SegmentAnalysisResult]: 分析结果列表
        """
        logger.info(f"批量分析 {len(segments)} 个片段")
        
        results = []
        total = len(segments)
        
        for i, segment in enumerate(segments):
            # 进度回调
            if progress_callback:
                progress_callback(i + 1, total)
            
            try:
                result = self.analyze_segment(
                    segment_path=segment["path"],
                    query=query
                )
                results.append(result)
                
                logger.debug(f"片段{i+1}/{total} 分析完成")
                
            except Exception as e:
                logger.error(f"片段{i+1}/{total} 分析失败：{e}")
                # 创建错误结果
                error_result = SegmentAnalysisResult(
                    segment_index=segment.get("index", i),
                    start_time=segment.get("start", 0.0),
                    end_time=segment.get("end", 0.0),
                    query=query,
                    has_relevant_content=False,
                    description=f"分析失败：{str(e)}",
                    confidence=0.0,
                    raw_response="",
                    metadata={"error": str(e)}
                )
                results.append(error_result)
        
        logger.info(f"批量分析完成，成功{len(results)}个结果")
        return results
    
    def _run_inference(
        self,
        video_path: str,
        prompt: str,
        **kwargs
    ) -> str:
        """
        执行单模态推理（仅视频）
        
        Args:
            video_path: 视频文件路径
            prompt: Prompt 文本
            
        Returns:
            str: 模型响应文本
        """
        # 检查模型是否已加载
        if self.model_loader.model is None:
            logger.info("模型未加载，自动加载默认模型...")
            self.model_loader.load_model("Qwen/Qwen3-Omni-30B-A3B-Thinking")
        
        # 使用 vLLM 进行推理
        try:
            from vllm import SamplingParams
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 应用聊天模板
            processor = self.model_loader.processor
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 设置采样参数
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", 0.6),
                top_p=kwargs.get("top_p", 0.95),
                top_k=kwargs.get("top_k", 20),
                max_tokens=kwargs.get("max_tokens", 2048)
            )
            
            # 执行生成
            outputs = self.model_loader.model.generate(
                [text],
                sampling_params=sampling_params
            )
            
            return outputs[0].outputs[0].text
            
        except Exception as e:
            logger.error(f"推理失败：{e}")
            raise RuntimeError(f"推理失败：{e}")
    
    def _run_multimodal_inference(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """
        执行多模态推理（视频 + 音频）
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            str: 模型响应文本
        """
        # 类似 _run_inference，但支持音频输入
        # 具体实现取决于模型的音频支持方式
        return self._run_inference(
            video_path=messages[0]["content"][0]["video"],
            prompt=messages[0]["content"][-1]["text"],
            **kwargs
        )
    
    def _parse_response(
        self,
        response: str,
        query: str
    ) -> Dict[str, Any]:
        """
        解析模型响应
        
        Args:
            response: 模型响应文本
            query: 原始查询
            
        Returns:
            Dict[str, Any]: 解析后的结果
        """
        # 尝试提取 JSON 格式响应
        json_match = re.search(r'\{[\s\S]*\}', response)
        
        if json_match:
            try:
                data = json.loads(json_match.group())
                return {
                    "has_relevant": data.get("has_relevant", False),
                    "description": data.get("description", response),
                    "confidence": data.get("confidence", 0.5),
                    "metadata": data
                }
            except json.JSONDecodeError:
                pass
        
        # 无法解析 JSON，使用启发式判断
        has_relevant = self._check_relevance(response, query)
        confidence = 0.7 if has_relevant else 0.3
        
        return {
            "has_relevant": has_relevant,
            "description": response.strip(),
            "confidence": confidence,
            "metadata": {}
        }
    
    def _check_relevance(self, response: str, query: str) -> bool:
        """
        启发式判断响应是否相关
        
        Args:
            response: 模型响应
            query: 原始查询
            
        Returns:
            bool: 是否相关
        """
        # 检查是否有否定词
        negative_patterns = [
            "没有", "未发现", "不存在", "没看到", "未出现",
            "not found", "no", "none", "didn't see"
        ]
        
        response_lower = response.lower()
        
        # 如果响应包含多个否定词，可能是不相关
        negative_count = sum(1 for pattern in negative_patterns if pattern in response_lower)
        
        if negative_count >= 2:
            return False
        
        # 如果响应长度过短（<10 字），可能无内容
        if len(response.strip()) < 10:
            return False
        
        return True
    
    def _parse_segment_path(self, path: str) -> Dict[str, Any]:
        """
        从片段路径提取信息
        
        假设格式：xxx_seg001_0000-0060.mp4
        
        Args:
            path: 文件路径
            
        Returns:
            Dict[str, Any]: {index, start, end}
        """
        filename = Path(path).stem
        
        # 尝试匹配模式：xxx_seg001_0000-0060
        match = re.search(r'seg(\d+)_(\d+)-(\d+)', filename)
        
        if match:
            return {
                "index": int(match.group(1)),
                "start": float(match.group(2)),
                "end": float(match.group(3))
            }
        
        return {"index": 0, "start": 0.0, "end": 0.0}
    
    def _build_multimodal_message(
        self,
        video_path: str,
        audio_path: str,
        query: str
    ) -> List[Dict[str, Any]]:
        """
        构建多模态消息
        
        Args:
            video_path: 视频路径
            audio_path: 音频路径
            query: 查询文本
            
        Returns:
            List[Dict[str, Any]]: 消息列表
        """
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": query}
                ]
            }
        ]


# 命令行工具
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="视频推理引擎")
    parser.add_argument("--video", "-v", type=str, required=True, help="视频文件")
    parser.add_argument("--query", "-q", type=str, required=True, help="查询内容")
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Thinking")
    parser.add_argument("--template", "-t", type=str, default="general", help="Prompt 模板")
    parser.add_argument("--output", "-o", type=str, help="输出 JSON 文件")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 创建组件
    loader = ModelLoader()
    engine = VideoInferenceEngine(loader, default_prompt_template=args.template)
    
    # 执行分析
    print(f"🧠 分析视频：{args.video}")
    print(f"🔍 查询：{args.query}")
    print(f"🧩 模型：{args.model}")
    
    result = engine.analyze_segment(args.video, args.query)
    
    # 输出结果
    print(f"\n📊 分析结果:")
    print(f"  相关性：{'是' if result.has_relevant_content else '否'}")
    print(f"  置信度：{result.confidence:.2f}")
    print(f"  描述：{result.description}")
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\n✅ 结果已保存：{args.output}")
