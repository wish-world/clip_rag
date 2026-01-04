from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
import uvicorn
import os
import sys
from dotenv import load_dotenv
import time
import uuid
import json
import asyncio

# 添加当前目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

load_dotenv()

# 全局变量
vector_manager = None
rag_graph = None
threads_store = {}
runs_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理"""
    global vector_manager, rag_graph
    
    print("启动CLIP RAG API服务...")
    
    try:
        # 先尝试直接导入
        from vector_store import VectorStoreManager
        from rag_graph import RAGGraph
        print("✓ 使用直接导入成功")
    except ImportError as e1:
        try:
            # 再尝试相对导入
            from .vector_store import VectorStoreManager
            from .rag_graph import RAGGraph
            print("✓ 使用相对导入成功")
        except ImportError as e2:
            print(f"✗ 导入失败:")
            print(f"  直接导入错误: {e1}")
            print(f"  相对导入错误: {e2}")
            print("当前目录文件:", [f for f in os.listdir(current_dir) if f.endswith('.py')])
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
        rag_graph = None
    
    yield
    
    print("关闭服务...")

# 创建 FastAPI 应用实例
app = FastAPI(title="CLIP RAG API", version="1.0.0", lifespan=lifespan)

# 添加 CORS 中间件
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

@app.post("/threads")
async def create_thread():
    """创建新的对话线程 - 完全符合LangGraph规范"""
    thread_id = str(uuid.uuid4())
    
    # 创建完全兼容的线程结构
    thread_data = {
        "thread_id": thread_id,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "assistant_id": "clip-rag-assistant",
        "values": {
            "messages": []
        },
        "metadata": {
            "created_by": "api",
            "assistant_id": "clip-rag-assistant"
        },
        "status": "active"
    }
    
    threads_store[thread_id] = thread_data
    print(f"✓ 创建新线程: {thread_id}")
    
    return thread_data

@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    """获取线程信息"""
    if thread_id not in threads_store:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    # 确保返回完整的线程结构
    thread = threads_store[thread_id]
    thread["updated_at"] = datetime.now().isoformat()
    return thread

# ========== 关键修复：完全兼容Agent Chat UI的SSE流 ==========
@app.post("/threads/{thread_id}/runs/stream")
async def create_thread_run_stream(thread_id: str, request: dict):
    """在特定线程中创建流式运行 - 完全兼容Agent Chat UI"""
    print(f"\n{'='*60}")
    print(f"接收到 /threads/{thread_id}/runs/stream 请求")
    print(f"请求时间: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    if not rag_graph:
        error_msg = "RAG服务未初始化。请先初始化向量存储。"
        print(f"✗ {error_msg}")
        raise HTTPException(status_code=503, detail=error_msg)
    
    # 解析请求消息
    messages = []
    
    # 处理不同的请求格式
    if isinstance(request, dict):
        if "input" in request and isinstance(request["input"], dict):
            # LangGraph标准格式
            messages = request["input"].get("messages", [])
            print("✓ 使用LangGraph标准格式 (input.messages)")
        elif "messages" in request:
            # 直接messages格式
            messages = request.get("messages", [])
            print("✓ 使用直接messages格式")
        else:
            print(f"⚠ 未知请求格式: {request.keys() if isinstance(request, dict) else type(request)}")
    
    print(f"✓ 解析到 {len(messages)} 条消息")
    
    # 获取最后一条用户消息
    last_user_message = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        
        # 检查消息类型
        is_human = msg.get("type") == "human" or msg.get("role") in ["user", "human"]
        
        if is_human:
            content = msg.get("content", "")
            if isinstance(content, str):
                last_user_message = content
            elif isinstance(content, list):
                # 处理复杂内容格式
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        last_user_message = item.get("text", "")
                        break
            break
    
    if not last_user_message:
        error_msg = "未找到有效的用户消息"
        print(f"✗ {error_msg}")
        print(f"✗ 可用消息: {messages}")
        raise HTTPException(status_code=400, detail=error_msg)
    
    print(f"✓ 用户消息: {last_user_message[:100]}...")
    
    # 获取回答
    try:
        print("⏳ 正在生成回答...")
        start_time = time.time()
        answer = await rag_graph.ainvoke(last_user_message)
        elapsed = time.time() - start_time
        print(f"✓ 生成回答成功 (耗时: {elapsed:.2f}秒, 长度: {len(answer)} 字符)")
    except Exception as e:
        error_msg = f"生成回答出错: {str(e)}"
        print(f"✗ {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)
    
    # 生成唯一ID
    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    ai_message_id = f"msg_ai_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    human_message_id = f"msg_human_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # 更新线程存储
    if thread_id not in threads_store:
        threads_store[thread_id] = {
            "thread_id": thread_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "values": {"messages": []},
            "metadata": {}
        }
    
    async def generate_compatible_sse():
        """生成完全兼容Agent Chat UI的SSE事件流"""
        
        # 关键：Agent Chat UI期望的事件序列和格式
        
        # 1. 运行开始事件
        run_start_event = {
            "event": "on_run_start",
            "run_id": run_id,
            "session_id": thread_id,
            "input": {
                "input": {
                    "messages": messages
                }
            },
            "tags": ["langchain:run_type:chain"],
            "metadata": {},
            "name": "langgraph",
            "start_time": int(time.time() * 1000)
        }
        yield f"data: {json.dumps(run_start_event, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.001)
        
        # 2. 链开始事件
        chain_start_event = {
            "event": "on_chain_start",
            "run_id": run_id,
            "parent_run_id": None,
            "tags": ["langchain:run_type:chain"],
            "metadata": {},
            "name": "langgraph",
            "data": {
                "input": {
                    "messages": messages
                }
            }
        }
        yield f"data: {json.dumps(chain_start_event, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.001)
        
        # 3. 聊天模型开始事件
        chat_start_event = {
            "event": "on_chat_model_start",
            "run_id": run_id,
            "name": "RAG_Chat_Model",
            "parent_run_id": None,
            "tags": ["langchain:run_type:llm"],
            "metadata": {
                "ls_model_name": "deepseek-chat",
                "ls_provider": "deepseek"
            },
            "data": {
                "input": {
                    "messages": messages
                }
            }
        }
        yield f"data: {json.dumps(chat_start_event, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.001)
        
        # 4. 关键修复：使用on_chain_stream事件，发送累积内容
        # Agent Chat UI期望on_chain_stream而不是on_chat_model_stream
        accumulated_content = ""
        
        # 分块发送，每次发送几个字符，模拟打字效果
        chunk_size = 2
        chunks = [answer[i:i+chunk_size] for i in range(0, len(answer), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            accumulated_content += chunk
            
            # 关键：使用on_chain_stream事件，包含完整的消息对象
            stream_event = {
                "event": "on_chain_stream",
                "run_id": run_id,
                "parent_run_id": None,
                "tags": ["langchain:run_type:chain"],
                "metadata": {},
                "data": {
                    "chunk": {
                        "output": {
                            "messages": [
                                {
                                    "content": accumulated_content,
                                    "type": "ai",
                                    "id": ai_message_id,
                                    "example": False,
                                    "additional_kwargs": {},
                                    "response_metadata": {},
                                    "tool_calls": [],
                                    "invalid_tool_calls": []
                                }
                            ]
                        }
                    }
                }
            }
            yield f"data: {json.dumps(stream_event, ensure_ascii=False)}\n\n"
            
            # 控制流式速度
            await asyncio.sleep(0.02)
        
        # 5. 聊天模型结束事件
        chat_end_event = {
            "event": "on_chat_model_end",
            "run_id": run_id,
            "name": "RAG_Chat_Model",
            "parent_run_id": None,
            "tags": ["langchain:run_type:llm"],
            "metadata": {
                "ls_model_name": "deepseek-chat",
                "ls_provider": "deepseek"
            },
            "data": {
                "output": {
                    "generations": [[{
                        "message": {
                            "content": answer,
                            "additional_kwargs": {},
                            "response_metadata": {
                                "model_name": "deepseek-chat",
                                "finish_reason": "stop"
                            },
                            "type": "AIMessage",
                            "id": ai_message_id,
                            "tool_calls": [],
                            "invalid_tool_calls": [],
                            "usage_metadata": None
                        },
                        "type": "ChatGeneration"
                    }]],
                    "llm_output": {
                        "model_name": "deepseek-chat"
                    }
                }
            },
            "status": "completed"
        }
        yield f"data: {json.dumps(chat_end_event, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.001)
        
        # 6. 链结束事件
        chain_end_event = {
            "event": "on_chain_end",
            "run_id": run_id,
            "parent_run_id": None,
            "tags": ["langchain:run_type:chain"],
            "metadata": {},
            "data": {
                "output": {
                    "messages": [
                        {
                            "content": answer,
                            "type": "ai",
                            "id": ai_message_id,
                            "example": False,
                            "additional_kwargs": {},
                            "response_metadata": {
                                "model_name": "deepseek-chat",
                                "finish_reason": "stop"
                            },
                            "tool_calls": [],
                            "invalid_tool_calls": []
                        }
                    ]
                }
            }
        }
        yield f"data: {json.dumps(chain_end_event, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.001)
        
        # 7. 运行结束事件
        run_end_event = {
            "event": "on_run_end",
            "run_id": run_id,
            "session_id": thread_id,
            "output": {
                "output": {
                    "messages": [
                        {
                            "content": answer,
                            "type": "ai",
                            "id": ai_message_id
                        }
                    ]
                }
            },
            "tags": ["langchain:run_type:chain"],
            "metadata": {}
        }
        yield f"data: {json.dumps(run_end_event, ensure_ascii=False)}\n\n"
        
        # 8. 发送空对象表示流结束（Agent Chat UI的关键要求）
        yield "data: {}\n\n"
    
    # 保存消息到线程存储
    thread = threads_store[thread_id]
    if "values" not in thread:
        thread["values"] = {"messages": []}
    elif "messages" not in thread["values"]:
        thread["values"]["messages"] = []
    
    # 添加人类消息
    thread["values"]["messages"].append({
        "type": "human",
        "content": last_user_message,
        "id": human_message_id,
        "example": False,
        "additional_kwargs": {}
    })
    
    # 添加AI消息
    thread["values"]["messages"].append({
        "type": "ai",
        "content": answer,
        "id": ai_message_id,
        "example": False,
        "additional_kwargs": {},
        "response_metadata": {
            "model_name": "deepseek-chat",
            "finish_reason": "stop"
        }
    })
    
    thread["updated_at"] = datetime.now().isoformat()
    
    # 响应头
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream; charset=utf-8",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*"
    }
    
    print(f"✓ 开始发送SSE流 (run_id: {run_id})")
    print(f"✓ 回答长度: {len(answer)} 字符")
    print(f"✓ 线程 {thread_id} 已更新")
    print(f"{'='*60}")
    
    return StreamingResponse(generate_compatible_sse(), media_type="text/event-stream", headers=headers)

@app.post("/threads/{thread_id}/history")
async def get_thread_history(thread_id: str, request: dict = None):
    """获取线程历史记录 - 完全兼容格式"""
    print(f"\n接收到 /threads/{thread_id}/history 请求")
    print(f"请求时间: {datetime.now().strftime('%H:%M:%S')}")
    
    if thread_id not in threads_store:
        print(f"✗ 线程不存在: {thread_id}")
        raise HTTPException(status_code=404, detail="Thread not found")
    
    thread = threads_store[thread_id]
    messages = thread.get("values", {}).get("messages", [])
    
    # 标准化消息格式
    standardized_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        
        # 确保每条消息都有必需字段
        standardized_msg = {
            "type": msg.get("type", "human" if msg.get("role") == "user" else "ai"),
            "content": msg.get("content", ""),
            "id": msg.get("id", str(uuid.uuid4())),
            "additional_kwargs": msg.get("additional_kwargs", {}),
            "example": msg.get("example", False),
            "response_metadata": msg.get("response_metadata", {}),
            "name": msg.get("name"),
            "tool_calls": msg.get("tool_calls", []),
            "invalid_tool_calls": msg.get("invalid_tool_calls", []),
            "usage_metadata": msg.get("usage_metadata")
        }
        
        standardized_messages.append(standardized_msg)
    
    print(f"✓ 返回 {len(standardized_messages)} 条历史记录")
    return standardized_messages

@app.get("/threads/{thread_id}/history")
async def get_thread_history_get(thread_id: str):
    """GET方法获取线程历史记录"""
    return await get_thread_history(thread_id, None)

# ========== 测试和调试端点 ==========

@app.get("/test-sse")
async def test_sse():
    """测试SSE流端点"""
    async def test_stream():
        test_message = "这是一个测试响应，用于验证SSE流是否正常工作。"
        
        # 运行开始
        yield f"data: {json.dumps({'event': 'on_run_start', 'run_id': 'test_run', 'name': 'test'})}\n\n"
        await asyncio.sleep(0.1)
        
        # 链开始
        yield f"data: {json.dumps({'event': 'on_chain_start', 'run_id': 'test_run', 'name': 'test'})}\n\n"
        await asyncio.sleep(0.1)
        
        # 聊天模型开始
        yield f"data: {json.dumps({'event': 'on_chat_model_start', 'run_id': 'test_run', 'name': 'test_model'})}\n\n"
        await asyncio.sleep(0.1)
        
        # 逐字符发送
        for i, char in enumerate(test_message):
            stream_event = {
                "event": "on_chain_stream",
                "run_id": "test_run",
                "data": {
                    "chunk": {
                        "output": {
                            "messages": [
                                {
                                    "content": test_message[:i+1],
                                    "type": "ai",
                                    "id": "test_msg"
                                }
                            ]
                        }
                    }
                }
            }
            yield f"data: {json.dumps(stream_event, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.05)
        
        # 聊天模型结束
        end_event = {
            "event": "on_chat_model_end",
            "run_id": "test_run",
            "name": "test_model",
            "data": {
                "output": {
                    "generations": [[{
                        "message": {
                            "content": test_message,
                            "type": "AIMessage",
                            "id": "test_msg"
                        }
                    }]]
                }
            }
        }
        yield f"data: {json.dumps(end_event, ensure_ascii=False)}\n\n"
        
        # 链结束
        yield f"data: {json.dumps({'event': 'on_chain_end', 'run_id': 'test_run'})}\n\n"
        
        # 运行结束
        yield f"data: {json.dumps({'event': 'on_run_end', 'run_id': 'test_run'})}\n\n"
        
        # 流结束
        yield "data: {}\n\n"
    
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream; charset=utf-8",
        "X-Accel-Buffering": "no"
    }
    
    return StreamingResponse(test_stream(), media_type="text/event-stream", headers=headers)

@app.get("/test-simple")
async def test_simple():
    """最简单的测试端点"""
    async def simple_stream():
        yield "data: 这是一条测试消息\n\n"
        yield "data: 这是第二条消息\n\n"
        yield "data: 这是最后一条消息\n\n"
        yield "data: {}\n\n"
    
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream; charset=utf-8",
        "X-Accel-Buffering": "no"
    }
    
    return StreamingResponse(simple_stream(), media_type="text/event-stream", headers=headers)

# ========== 原有接口 ==========

@app.get("/")
async def root():
    return {
        "message": "CLIP RAG API 服务运行中", 
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "测试端点": {
                "test_sse": "GET /test-sse",
                "test_simple": "GET /test-simple"
            },
            "LangGraph接口": {
                "info": "GET /info",
                "create_thread": "POST /threads",
                "get_thread": "GET /threads/{thread_id}",
                "thread_run_stream": "POST /threads/{thread_id}/runs/stream",
                "thread_history": {
                    "methods": ["GET", "POST"],
                    "path": "/threads/{thread_id}/history"
                }
            },
            "原始接口": {
                "chat": "POST /chat",
                "status": "GET /status",
                "health": "GET /health"
            }
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

if __name__ == "__main__":
    # 运行应用
    print("\n" + "="*60)
    print("CLIP RAG 服务器启动")
    print("="*60)
    print(f"工作目录: {os.getcwd()}")
    print(f"API地址: http://localhost:8000")
    print(f"API文档: http://localhost:8000/docs")
    print(f"测试端点:")
    print(f"  - http://localhost:8000/test-sse")
    print(f"  - http://localhost:8000/test-simple")
    print("="*60)
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)