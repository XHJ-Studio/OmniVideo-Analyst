#!/usr/bin/env python3
"""
OmniVideo-Analyst API 服务

启动服务:
    python src/api.py --port 8000

API 端点:
    POST /analyze - 分析视频
    GET  /health  - 健康检查
"""

import argparse
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    """分析请求"""
    video_path: str
    query: str
    segment_length: int = 60
    model: str = "Qwen/Qwen3-Omni-30B-A3B-Thinking"


class AnalyzeResponse(BaseModel):
    """分析响应"""
    status: str
    video_path: str
    query: str
    segments: list
    summary: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    logger.info("启动 OmniVideo-Analyst API 服务...")
    # TODO: 初始化模型
    yield
    # 关闭时清理资源
    logger.info("关闭服务...")


app = FastAPI(
    title="OmniVideo-Analyst API",
    description="使用自然语言分析视频内容的 API 服务",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "OmniVideo-Analyst"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_video(request: AnalyzeRequest):
    """
    分析视频
    
    Args:
        request: 分析请求
        
    Returns:
        分析结果
    """
    # 验证视频文件
    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=400, detail=f"视频文件不存在：{video_path}")
    
    logger.info(f"收到分析请求：{request.query}")
    
    # TODO: 调用核心分析逻辑
    # from src.analyst import VideoAnalyst
    # analyst = VideoAnalyst(request.model)
    # result = analyst.analyze(request.video_path, request.query)
    
    # 示例响应
    return AnalyzeResponse(
        status="success",
        video_path=request.video_path,
        query=request.query,
        segments=[
            {
                "start_time": "00:00:00",
                "end_time": "00:01:00",
                "description": "示例片段",
                "relevance": 0.85
            }
        ],
        summary="分析完成"
    )


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "OmniVideo-Analyst API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


def main():
    parser = argparse.ArgumentParser(description="OmniVideo-Analyst API 服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="开发模式：自动重载")
    
    args = parser.parse_args()
    
    import uvicorn
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
