#!/usr/bin/env python3
"""
API 客户端示例

用法:
    1. 启动 API 服务：python src/api.py --port 8000
    2. 运行客户端：python examples/api_client.py --video path/to/video.mp4 --query "找红色自行车"
"""

import argparse
import time
import requests
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def submit_analysis(base_url: str, video_path: str, query: str) -> str:
    """提交分析任务"""
    response = requests.post(
        f"{base_url}/analyze",
        json={
            "video_path": video_path,
            "query": query,
            "segment_length": 60
        }
    )
    
    if response.status_code != 200:
        raise Exception(f"提交失败：{response.text}")
    
    data = response.json()
    return data["task_id"]


def wait_for_completion(base_url: str, task_id: str, interval: int = 5):
    """等待任务完成"""
    print(f"⏳ 等待任务完成...")
    
    while True:
        response = requests.get(f"{base_url}/tasks/{task_id}")
        
        if response.status_code != 200:
            raise Exception(f"查询失败：{response.text}")
        
        data = response.json()
        status = data["status"]
        progress = data.get("progress", 0) * 100
        message = data.get("message", "")
        
        print(f"  进度：{progress:.0f}% - {message}")
        
        if status in ["completed", "failed", "cancelled"]:
            break
        
        time.sleep(interval)
    
    return data


def main():
    parser = argparse.ArgumentParser(description="API 客户端示例")
    parser.add_argument("--video", "-v", type=str, required=True, help="视频文件路径")
    parser.add_argument("--query", "-q", type=str, required=True, help="查询内容")
    parser.add_argument("--server", "-s", type=str, default="http://localhost:8000", help="API 服务器地址")
    parser.add_argument("--output", "-o", type=str, default="api_report.json", help="输出报告路径")
    
    args = parser.parse_args()
    
    print(f"🎬 OmniVideo-Analyst API 客户端")
    print(f"=" * 50)
    print(f"📹 视频：{args.video}")
    print(f"🔍 查询：{args.query}")
    print(f"🌐 服务器：{args.server}")
    print(f"=" * 50)
    
    # 检查服务器
    try:
        response = requests.get(f"{args.server}/health")
        if response.status_code != 200:
            print(f"❌ 服务器未响应")
            sys.exit(1)
        print(f"✅ 服务器连接正常")
    except Exception as e:
        print(f"❌ 无法连接服务器：{e}")
        sys.exit(1)
    
    # 提交任务
    print(f"\n📤 提交任务...")
    task_id = submit_analysis(args.server, args.video, args.query)
    print(f"✅ 任务 ID: {task_id}")
    
    # 等待完成
    print(f"\n🔄 分析中...")
    result = wait_for_completion(args.server, task_id)
    
    # 显示结果
    print(f"\n{'=' * 50}")
    print(f"📊 分析结果")
    print(f"{'=' * 50}")
    print(f"状态：{result['status']}")
    print(f"进度：{result.get('progress', 0) * 100:.0f}%")
    
    if result["status"] == "completed" and result.get("result"):
        report = result["result"]
        print(f"\n📹 视频：{report.get('video_path', 'N/A')}")
        print(f"🔍 查询：{report.get('query', 'N/A')}")
        print(f"📋 相关片段：{report.get('relevant_segments', 0)}/{report.get('analyzed_segments', 0)}")
        print(f"📈 平均置信度：{report.get('statistics', {}).get('average_confidence', 0):.2f}")
        print(f"\n📝 总结：{report.get('summary', 'N/A')}")
        
        # 保存报告
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 报告已保存：{args.output}")
    
    elif result["status"] == "failed":
        print(f"\n❌ 分析失败：{result.get('error', '未知错误')}")
    
    elif result["status"] == "cancelled":
        print(f"\n⚠️  任务已取消")


if __name__ == "__main__":
    main()
