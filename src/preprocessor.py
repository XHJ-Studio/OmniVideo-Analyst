#!/usr/bin/env python3
"""
视频预处理模块

功能:
- 视频文件加载与验证
- 视频切段（按时间长度）
- 关键帧提取（场景变化检测）
- 音频轨道提取
- 格式转换（统一编码）

使用示例:
    from src.preprocessor import VideoPreprocessor
    
    preprocessor = VideoPreprocessor()
    segments = preprocessor.segment_video("input.mp4", segment_length=60)
    keyframes = preprocessor.extract_keyframes("input.mp4")
    audio = preprocessor.extract_audio("input.mp4")
"""

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import timedelta

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """视频信息"""
    path: str
    width: int
    height: int
    fps: float
    duration: float  # 秒
    frame_count: int
    codec: str
    has_audio: bool
    
    def __str__(self) -> str:
        return (f"VideoInfo({Path(self.path).name}, "
                f"{self.width}x{self.height}, "
                f"{self.fps}fps, "
                f"{timedelta(seconds=int(self.duration))})")


@dataclass
class VideoSegment:
    """视频片段信息"""
    index: int
    start_time: float  # 秒
    end_time: float    # 秒
    path: str
    frame_count: int


class VideoPreprocessor:
    """视频预处理器"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化预处理器
        
        Args:
            output_dir: 输出目录，默认为临时目录
        """
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/omnivideo")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"预处理器初始化完成，输出目录：{self.output_dir}")
    
    def get_video_info(self, video_path: str) -> VideoInfo:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            VideoInfo: 视频信息
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 无效的视频文件
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"视频文件不存在：{video_path}")
        
        # 使用 OpenCV 获取视频信息
        cap = cv2.VideoCapture(str(path))
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件：{video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # 获取编码器信息
        codec_code = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = self._fourcc_to_string(codec_code)
        
        # 检查是否有音频（使用 ffprobe）
        has_audio = self._check_has_audio(str(path))
        
        cap.release()
        
        info = VideoInfo(
            path=str(path),
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            frame_count=frame_count,
            codec=codec,
            has_audio=has_audio
        )
        
        logger.info(f"视频信息：{info}")
        return info
    
    def segment_video(
        self,
        video_path: str,
        segment_length: int = 60,
        overlap: int = 0,
        output_prefix: Optional[str] = None
    ) -> List[VideoSegment]:
        """
        将视频切分成多个片段
        
        Args:
            video_path: 视频文件路径
            segment_length: 每段长度（秒），默认 60 秒
            overlap: 段间重叠时间（秒），默认 0
            output_prefix: 输出文件前缀，默认使用视频名
            
        Returns:
            List[VideoSegment]: 视频片段列表
            
        Raises:
            FileNotFoundError: 文件不存在
        """
        # 获取视频信息
        info = self.get_video_info(video_path)
        path = Path(video_path)
        
        if output_prefix is None:
            output_prefix = path.stem
        
        segments = []
        current_time = 0.0
        segment_index = 0
        
        logger.info(f"开始切分视频：{info}")
        logger.info(f"切分参数：每段{segment_length}秒，重叠{overlap}秒")
        
        while current_time < info.duration:
            # 计算结束时间
            end_time = min(current_time + segment_length, info.duration)
            
            # 创建输出文件名
            output_filename = f"{output_prefix}_seg{segment_index:03d}_{int(current_time):04d}-{int(end_time):04d}.mp4"
            output_path = self.output_dir / output_filename
            
            # 使用 ffmpeg 切分视频
            self._cut_video_segment(
                str(path),
                str(output_path),
                current_time,
                end_time
            )
            
            # 计算片段帧数
            segment_duration = end_time - current_time
            segment_frames = int(segment_duration * info.fps)
            
            segment = VideoSegment(
                index=segment_index,
                start_time=current_time,
                end_time=end_time,
                path=str(output_path),
                frame_count=segment_frames
            )
            
            segments.append(segment)
            logger.info(f"切分完成：片段{segment_index} [{current_time:.1f}s - {end_time:.1f}s]")
            
            # 更新当前时间（考虑重叠）
            current_time = end_time - overlap
            segment_index += 1
            
            # 防止死循环
            if current_time >= info.duration:
                break
        
        logger.info(f"切分完成，共{len(segments)}个片段")
        return segments
    
    def extract_keyframes(
        self,
        video_path: str,
        threshold: float = 30.0,
        output_prefix: Optional[str] = None
    ) -> List[str]:
        """
        提取关键帧（基于场景变化检测）
        
        Args:
            video_path: 视频文件路径
            threshold: 场景变化阈值，默认 30.0
            output_prefix: 输出文件前缀
            
        Returns:
            List[str]: 关键帧文件路径列表
        """
        path = Path(video_path)
        if output_prefix is None:
            output_prefix = path.stem
        
        logger.info(f"开始提取关键帧：{path.name}, 阈值={threshold}")
        
        cap = cv2.VideoCapture(str(path))
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件：{video_path}")
        
        keyframes = []
        prev_frame = None
        frame_index = 0
        keyframe_index = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 检测场景变化
            if prev_frame is not None:
                # 计算帧差异
                diff = cv2.absdiff(prev_frame, gray)
                score = np.sum(diff) / diff.size
                
                # 如果差异超过阈值，认为是关键帧
                if score > threshold:
                    output_filename = f"{output_prefix}_keyframe{keyframe_index:03d}_f{frame_index:05d}.jpg"
                    output_path = self.output_dir / output_filename
                    
                    cv2.imwrite(str(output_path), frame)
                    keyframes.append(str(output_path))
                    
                    logger.debug(f"检测到关键帧：{frame_index}, 差异分数={score:.2f}")
                    keyframe_index += 1
            
            prev_frame = gray
            frame_index += 1
            
            # 进度提示（每 100 帧）
            if frame_index % 100 == 0:
                logger.debug(f"已处理{frame_index}帧，提取{keyframe_index}个关键帧")
        
        cap.release()
        
        logger.info(f"关键帧提取完成：共{keyframe_index}个关键帧")
        return keyframes
    
    def extract_audio(
        self,
        video_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        提取音频轨道
        
        Args:
            video_path: 视频文件路径
            output_path: 输出文件路径，默认自动生成
            
        Returns:
            str: 音频文件路径
        """
        path = Path(video_path)
        
        if output_path is None:
            output_filename = f"{path.stem}_audio.wav"
            output_path = str(self.output_dir / output_filename)
        
        logger.info(f"开始提取音频：{path.name}")
        
        # 使用 ffmpeg 提取音频
        cmd = [
            "ffmpeg", "-i", str(path),
            "-vn",  # 不要视频
            "-acodec", "pcm_s16le",  # PCM 编码
            "-ar", "16000",  # 采样率 16kHz
            "-ac", "1",  # 单声道
            "-y",  # 覆盖输出文件
            output_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"音频提取完成：{output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"音频提取失败：{e.stderr}")
            raise RuntimeError(f"音频提取失败：{e.stderr}")
    
    def cleanup(self, remove_all: bool = False):
        """
        清理临时文件
        
        Args:
            remove_all: 是否删除所有输出文件
        """
        if remove_all and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            logger.info(f"已清理输出目录：{self.output_dir}")
    
    def _fourcc_to_string(self, codec: int) -> str:
        """将 FourCC 代码转换为字符串"""
        return "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
    
    def _check_has_audio(self, video_path: str) -> bool:
        """检查视频是否有音频轨道"""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False
    
    def _cut_video_segment(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float
    ):
        """
        切分视频片段
        
        使用 ffmpeg 的 -ss 和 -to 参数进行精确切分
        """
        duration = end_time - start_time
        
        cmd = [
            "ffmpeg", "-i", input_path,
            "-ss", str(start_time),
            "-t", str(duration),
            "-c:v", "libx264",  # 重新编码以保证兼容性
            "-c:a", "aac",
            "-preset", "fast",  # 快速编码
            "-y",  # 覆盖输出文件
            output_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"视频切分失败：{e.stderr}")
            raise RuntimeError(f"视频切分失败：{e.stderr}")


# 命令行工具
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="视频预处理工具")
    parser.add_argument("--video", "-v", type=str, required=True, help="输入视频文件")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出目录")
    parser.add_argument("--segment-length", type=int, default=60, help="切段长度（秒）")
    parser.add_argument("--keyframes", action="store_true", help="提取关键帧")
    parser.add_argument("--audio", action="store_true", help="提取音频")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 创建预处理器
    preprocessor = VideoPreprocessor(args.output)
    
    # 获取视频信息
    info = preprocessor.get_video_info(args.video)
    print(f"\n📹 视频信息:")
    print(f"  文件：{info.path}")
    print(f"  分辨率：{info.width}x{info.height}")
    print(f"  帧率：{info.fps}fps")
    print(f"  时长：{timedelta(seconds=int(info.duration))}")
    print(f"  编码：{info.codec}")
    print(f"  音频：{'有' if info.has_audio else '无'}")
    
    # 切分视频
    print(f"\n✂️  切分视频...")
    segments = preprocessor.segment_video(args.video, args.segment_length)
    print(f"  共切分为 {len(segments)} 个片段")
    
    # 提取关键帧
    if args.keyframes:
        print(f"\n🖼️  提取关键帧...")
        keyframes = preprocessor.extract_keyframes(args.video)
        print(f"  共提取 {len(keyframes)} 个关键帧")
    
    # 提取音频
    if args.audio:
        print(f"\n🎵 提取音频...")
        audio_path = preprocessor.extract_audio(args.video)
        print(f"  音频文件：{audio_path}")
    
    print(f"\n✅ 预处理完成！输出目录：{preprocessor.output_dir}")
