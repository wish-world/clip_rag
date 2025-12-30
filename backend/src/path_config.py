import os
from pathlib import Path

def get_project_root():
    """获取项目根目录"""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    return project_root

def get_documents_dir():
    """获取文档目录"""
    project_root = get_project_root()
    return project_root / "documents"

def get_vectorstore_dir():
    """获取向量存储目录"""
    project_root = get_project_root()
    return project_root / "vectorstore"

def get_frontend_dir():
    """获取前端目录"""
    project_root = get_project_root()
    return project_root / "frontend"

# 测试
if __name__ == "__main__":
    print("项目根目录:", get_project_root())
    print("文档目录:", get_documents_dir())
    print("向量存储目录:", get_vectorstore_dir())
    print("前端目录:", get_frontend_dir())