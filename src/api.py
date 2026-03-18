#!/usr/bin/env python3
"""
OmniVideo-Analyst API 服务（完整版）

功能:
- 异步任务队列
- 视频分析 API
- 进度查询端点
- 文件上传支持
- WebSocket 实时推送

启动服务:
    python src/api.py --port 8000

API 端点:
    POST   /analyze         - 提交分析任务
    GET    /tasks/{id}      - 查询任务状态
    GET    /tasks/{id}/ws   - WebSocket 实时推送
    POST   /upload          - 上传视频文件
    GET    /health          - 健康检查
    DELETE /tasks/{id}      - 取消任务
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 导入核心模块
from .preprocessor import VideoPreprocessor
from .model_loader import ModelLoader
from .inference import VideoInferenceEngine, SegmentAnalysisResult
from .aggregator import ResultAggregator, AnalysisReport

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ========== 数据模型 ==========

class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalyzeRequest(BaseModel):
    """分析请求"""
    video_path: str
    query: str
    segment_length: int = 60
    model: str = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
    tensor_parallel_size: int = 1
    confidence_threshold: float = 0.5
    output_format: str = "json"


class AnalyzeResponse(BaseModel):
    """分析响应"""
    task_id: str
    status: TaskStatus
    message: str
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0  # 0-1


class TaskInfo(BaseModel):
    """任务信息"""
    task_id: str
    status: TaskStatus
    query: str
    video_path: str
    created_at: str
    completed_at: Optional[str] = None
    progress: float = 0.0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ========== 任务管理器 ==========

class TaskManager:
    """任务管理器（内存存储）"""
    
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.websocket_connections: Dict[str, List[WebSocket]] = {}
    
    def create_task(self, video_path: str, query: str) -> str:
        """创建任务"""
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "task_id": task_id,
            "status": TaskStatus.PENDING,
            "query": query,
            "video_path": video_path,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "progress": 0.0,
            "message": "任务已创建",
            "result": None,
            "error": None
        }
        logger.info(f"创建任务：{task_id}")
        return task_id
    
    def update_task(self, task_id: str, **kwargs):
        """更新任务状态"""
        if task_id not in self.tasks:
            raise ValueError(f"任务不存在：{task_id}")
        
        self.tasks[task_id].update(kwargs)
        logger.debug(f"更新任务 {task_id}: {kwargs}")
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
        return self.tasks.get(task_id)
    
    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Dict[str, Any]]:
        """列出任务"""
        tasks = list(self.tasks.values())
        if status:
            tasks = [t for t in tasks if t["status"] == status]
        return tasks
    
    def delete_task(self, task_id: str):
        """删除任务"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            logger.info(f"删除任务：{task_id}")
    
    def register_websocket(self, task_id: str, websocket: WebSocket):
        """注册 WebSocket 连接"""
        if task_id not in self.websocket_connections:
            self.websocket_connections[task_id] = []
        self.websocket_connections[task_id].append(websocket)
    
    async def broadcast_progress(self, task_id: str, progress: float, message: str):
        """广播进度更新"""
        if task_id in self.websocket_connections:
            disconnected = []
            for ws in self.websocket_connections[task_id]:
                try:
                    await ws.send_json({
                        "task_id": task_id,
                        "progress": progress,
                        "message": message
                    })
                except:
                    disconnected.append(ws)
            
            # 清理断开的连接
            for ws in disconnected:
                self.websocket_connections[task_id].remove(ws)


# 全局任务管理器
task_manager = TaskManager()


# ========== 分析任务执行器 ==========

async def execute_analysis_task(task_id: str, request: AnalyzeRequest):
    """执行分析任务"""
    try:
        logger.info(f"开始执行任务 {task_id}")
        task_manager.update_task(
            task_id,
            status=TaskStatus.PROCESSING,
            progress=0.1,
            message="初始化预处理..."
        )
        
        # Phase 1: 视频预处理
        preprocessor = VideoPreprocessor()
        video_info = preprocessor.get_video_info(request.video_path)
        
        task_manager.update_task(
            task_id,
            progress=0.2,
            message=f"视频信息：{video_info.width}x{video_info.height}, {video_info.duration:.1f}s"
        )
        
        segments = preprocessor.segment_video(
            request.video_path,
            segment_length=request.segment_length
        )
        
        task_manager.update_task(
            task_id,
            progress=0.3,
            message=f"切分完成：{len(segments)}个片段"
        )
        
        # Phase 2: 模型加载
        task_manager.update_task(
            task_id,
            progress=0.4,
            message="加载模型..."
        )
        
        model_loader = ModelLoader()
        model, processor = model_loader.load_model(
            request.model,
            tensor_parallel_size=request.tensor_parallel_size
        )
        
        task_manager.update_task(
            task_id,
            progress=0.5,
            message="模型加载完成"
        )
        
        # Phase 3: 视频推理
        engine = VideoInferenceEngine(model_loader)
        results = []
        total_segments = len(segments)
        
        for i, segment in enumerate(segments):
            result = engine.analyze_segment(segment.path, request.query)
            results.append(result)
            
            # 更新进度
            progress = 0.5 + 0.3 * (i + 1) / total_segments
            task_manager.update_task(
                task_id,
                progress=progress,
                message=f"分析中：{i+1}/{total_segments}"
            )
        
        task_manager.update_task(
            task_id,
            progress=0.8,
            message="分析完成，聚合结果..."
        )
        
        # Phase 4: 结果聚合
        aggregator = ResultAggregator(
            confidence_threshold=request.confidence_threshold
        )
        
        report = aggregator.aggregate(
            results,
            video_path=request.video_path,
            video_duration=video_info.duration
        )
        
        # 完成任务
        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=1.0,
            message="分析完成",
            completed_at=datetime.now().isoformat(),
            result=report.to_dict()
        )
        
        logger.info(f"任务 {task_id} 完成")
        
    except Exception as e:
        logger.exception(f"任务 {task_id} 失败：{e}")
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            error=str(e),
            message=f"分析失败：{str(e)}"
        )


# ========== FastAPI 应用 ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    logger.info("🚀 OmniVideo-Analyst API 启动...")
    yield
    logger.info("👋 OmniVideo-Analyst API 关闭...")


app = FastAPI(
    title="OmniVideo-Analyst API",
    description="使用自然语言分析视频内容的 API 服务",
    version="0.1.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "OmniVideo-Analyst",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def submit_analysis(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    提交视频分析任务
    
    Args:
        request: 分析请求
        background_tasks: 后台任务
        
    Returns:
        任务信息
    """
    # 验证视频文件
    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=400, detail=f"视频文件不存在：{video_path}")
    
    # 创建任务
    task_id = task_manager.create_task(request.video_path, request.query)
    
    # 添加到后台任务
    background_tasks.add_task(execute_analysis_task, task_id, request)
    
    return AnalyzeResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="任务已提交，正在处理中",
        created_at=datetime.now().isoformat()
    )


@app.get("/tasks", response_model=List[TaskInfo])
async def list_tasks(status: Optional[TaskStatus] = None):
    """列出所有任务"""
    tasks = task_manager.list_tasks(status)
    return [TaskInfo(**t) for t in tasks]


@app.get("/tasks/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str):
    """查询任务状态"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务不存在：{task_id}")
    return TaskInfo(**task)


@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """取消任务"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务不存在：{task_id}")
    
    if task["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
        raise HTTPException(status_code=400, detail="已完成或失败的任务无法取消")
    
    task_manager.update_task(
        task_id,
        status=TaskStatus.CANCELLED,
        message="任务已取消"
    )
    
    return {"message": f"任务 {task_id} 已取消"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    上传视频文件
    
    Args:
        file: 视频文件
        
    Returns:
        文件路径
    """
    # 验证文件类型
    allowed_types = ["video/mp4", "video/x-msvideo", "video/quicktime", "video/x-matroska"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型：{file.content_type}，支持：{allowed_types}"
        )
    
    # 保存文件
    upload_dir = Path("/tmp/omnivideo/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    logger.info(f"文件上传完成：{file_path}")
    
    return {
        "file_path": str(file_path),
        "filename": file.filename,
        "size": len(content)
    }


@app.websocket("/ws/tasks/{task_id}")
async def websocket_task_progress(websocket: WebSocket, task_id: str):
    """WebSocket 实时推送任务进度"""
    await websocket.accept()
    task_manager.register_websocket(task_id, websocket)
    
    try:
        # 发送当前状态
        task = task_manager.get_task(task_id)
        if task:
            await websocket.send_json({
                "task_id": task_id,
                "status": task["status"],
                "progress": task["progress"],
                "message": task["message"]
            })
        
        # 保持连接
        while True:
            await websocket.receive_text()
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket 断开：{task_id}")
    except Exception as e:
        logger.error(f"WebSocket 错误：{e}")


# ========== 命令行入口 ==========

def main():
    import argparse
    
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
