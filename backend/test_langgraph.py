import requests
import json

def test_langgraph_api():
    """测试LangGraph兼容接口"""
    print("测试LangGraph兼容接口...")
    
    base_url = "http://localhost:8000"
    
    # 测试/info接口
    print("\n1. 测试 /info 接口:")
    resp = requests.get(f"{base_url}/info")
    print(f"   状态码: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"   助手列表: {data.get('assistants')}")
        print(f"   图列表: {data.get('graphs')}")
    else:
        print(f"   响应: {resp.text}")
    
    # 测试/assistants接口
    print("\n2. 测试 /assistants/clip-rag-assistant 接口:")
    resp = requests.get(f"{base_url}/assistants/clip-rag-assistant")
    print(f"   状态码: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"   助手ID: {data.get('assistant_id')}")
        print(f"   助手名称: {data.get('name')}")
    else:
        print(f"   响应: {resp.text}")
    
    # 测试运行接口
    print("\n3. 测试 /assistants/clip-rag-assistant/runs 接口:")
    payload = {
        "configurable": {
            "thread_id": "test-thread-123",
            "assistant_id": "clip-rag-assistant"
        },
        "messages": [
            {
                "role": "user",
                "content": "什么是RAG？"
            }
        ]
    }
    
    resp = requests.post(
        f"{base_url}/assistants/clip-rag-assistant/runs",
        json=payload
    )
    
    print(f"   状态码: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"   运行ID: {data.get('run_id')}")
        print(f"   状态: {data.get('status')}")
        if data.get('messages'):
            print(f"   回答: {data['messages'][0].get('content')[:100]}...")
    else:
        print(f"   响应: {resp.text}")
    
    print("\n✅ LangGraph兼容接口测试完成！")

if __name__ == "__main__":
    test_langgraph_api()