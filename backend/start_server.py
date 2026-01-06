#!/usr/bin/env python3
"""
CLIP RAG 多模态服务器启动脚本
"""

import os
import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print("=" * 60)
print("CLIP RAG 多模态服务器启动")
print("=" * 60)
print(f"工作目录: {current_dir}")

try:
    from src.main import app
    import uvicorn
    print("✓ 导入成功")
except ImportError as e:
    print(f"✗✗ 导入失败: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("启动多模态RAG服务器...")
    print("=" * 60)
    print("API地址: http://localhost:8000")
    print("API文档: http://localhost:8000/docs")
    print("健康检查: http://localhost:8000/health")
    print("状态检查: http://localhost:8000/status")
    print("多模态上传: POST /multimodal/upload")
    print("LangGraph信息: http://localhost:8000/info")
    print("\n按 Ctrl+C 停止服务器")
    print("=" * 60)
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )