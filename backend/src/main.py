from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime
import uvicorn
import os
from dotenv import load_dotenv
import time

load_dotenv()

vector_manager = None
rag_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理"""
    global vector_manager, rag_graph
    
    print("启动CLIP RAG API服务...")
    
    try:
        from .vector_store import VectorStoreManager
        from .rag_graph import RAGGraph
    except ImportError as e:
        print(f"✗ 导入本地模块失败: {e}")
        yield
        return
    
    print("初始化向量存储管理器...")
    vector_manager = VectorStoreManager()
    
    print("初始化向量存储...")
    is_initialized = vector_manager.initialize_vectorstore()
    
    if is_initialized and vector_manager.vectorstore:
        print("初始化RAG图...")
        try:
            retriever = vector_manager.as_retriever()
            rag_graph = RAGGraph(retriever)
            print("✓ RAG服务初始化完成")
        except Exception as e:
            print(f"✗ 初始化RAG图失败: {e}")
            rag_graph = None
    else:
        print("⚠ 向量存储未初始化，RAG服务将不可用")
        print("请先运行 python initialize_vectorstore.py 初始化向量存储")
        rag_graph = None
    
    yield
    
    print("关闭服务...")

app = FastAPI(title="CLIP RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求/响应模型
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    assistant_id: Optional[str] = "clip-rag-assistant"

class ChatResponse(BaseModel):
    messages: List[Dict[str, Any]]
    thread_id: str
    assistant_id: str

class StatusResponse(BaseModel):
    vectorstore_initialized: bool
    rag_graph_initialized: bool
    message: str
    vectorstore_path: Optional[str]
    document_count: Optional[int]

class AssistantConfig(BaseModel):
    configurable: Dict[str, str]

class LangGraphRunRequest(BaseModel):
    configurable: Dict[str, str]
    messages: List[Dict[str, Any]]

# ========== LangGraph兼容接口 ==========

@app.get("/info")
async def langgraph_info():
    """LangGraph兼容的info端点"""
    return {
        "assistants": ["clip-rag-assistant"],
        "graphs": ["clip-rag-graph"],
        "config_schema": {
            "type": "object",
            "properties": {
                "assistant_id": {
                    "type": "string",
                    "default": "clip-rag-assistant"
                },
                "thread_id": {
                    "type": "string"
                }
            }
        }
    }

@app.get("/assistants/{assistant_id}")
async def get_assistant(assistant_id: str):
    """获取助手信息 - LangGraph兼容接口"""
    if assistant_id != "clip-rag-assistant":
        raise HTTPException(status_code=404, detail="Assistant not found")
    
    return {
        "assistant_id": assistant_id,
        "config": {
            "configurable": {
                "thread_id": "default",
                "assistant_id": assistant_id
            }
        },
        "updated_at": datetime.now().isoformat(),
        "created_at": datetime.now().isoformat(),
        "name": "CLIP RAG Assistant",
        "description": "基于DeepSeek的RAG文档助手"
    }

@app.post("/assistants/{assistant_id}/runs")
async def create_run(assistant_id: str, request: LangGraphRunRequest):
    """创建运行 - LangGraph兼容接口"""
    if assistant_id != "clip-rag-assistant":
        raise HTTPException(status_code=404, detail="Assistant not found")
    
    if not rag_graph:
        raise HTTPException(
            status_code=503, 
            detail="RAG服务未初始化。请先初始化向量存储。"
        )
    
    thread_id = request.configurable.get("thread_id", f"thread_{int(time.time())}")
    messages = request.messages
    
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    # 获取最后一条用户消息
    last_user_message = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_message = msg.get("content")
            break
    
    if not last_user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # 获取回答
    try:
        answer = await rag_graph.ainvoke(last_user_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "run_id": f"run_{int(time.time())}",
        "thread_id": thread_id,
        "assistant_id": assistant_id,
        "status": "completed",
        "messages": [
            {
                "role": "assistant",
                "content": answer
            }
        ]
    }

# ========== 原有接口 ==========

@app.get("/")
async def root():
    return {
        "message": "CLIP RAG API 服务运行中", 
        "status": "healthy",
        "endpoints": {
            "chat": "POST /chat",
            "status": "GET /status",
            "health": "GET /health",
            "langgraph_info": "GET /info",
            "langgraph_assistant": "GET /assistants/{assistant_id}",
            "langgraph_run": "POST /assistants/{assistant_id}/runs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """聊天端点"""
    try:
        if not rag_graph:
            raise HTTPException(
                status_code=503, 
                detail="RAG服务未初始化。请先初始化向量存储。"
            )
        
        answer = await rag_graph.ainvoke(request.message)
        
        thread_id = request.thread_id or f"thread_{int(datetime.now().timestamp())}"
        
        response = {
            "messages": [
                {
                    "role": "assistant",
                    "content": answer
                }
            ],
            "thread_id": thread_id,
            "assistant_id": request.assistant_id or "clip-rag-assistant"
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """获取服务状态"""
    vectorstore_initialized = vector_manager.vectorstore is not None if vector_manager else False
    rag_graph_initialized = rag_graph is not None
    
    message = "服务运行正常"
    if not vectorstore_initialized:
        message = "向量存储未初始化"
    elif not rag_graph_initialized:
        message = "RAG图未初始化"
    
    document_count = None
    if vector_manager and vector_manager.vectorstore:
        try:
            document_count = vector_manager.vectorstore._collection.count()
        except:
            pass
    
    return {
        "vectorstore_initialized": vectorstore_initialized,
        "rag_graph_initialized": rag_graph_initialized,
        "message": message,
        "vectorstore_path": vector_manager.vectorstore_dir if vector_manager else None,
        "document_count": document_count
    }