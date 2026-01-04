import requests
import json
import uuid

def test_all_endpoints():
    print("测试所有LangGraph兼容端点...")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # 1. 测试/threads端点
    print("1. 测试 POST /threads:")
    response = requests.post(f"{base_url}/threads")
    if response.status_code == 200:
        thread_data = response.json()
        thread_id = thread_data.get("thread_id")
        print(f"   ✅ 成功创建线程: {thread_id}")
    else:
        print(f"   ❌ 创建线程失败: {response.status_code}")
        return
    
    # 2. 测试/threads/{thread_id}端点
    print(f"\n2. 测试 GET /threads/{thread_id}:")
    response = requests.get(f"{base_url}/threads/{thread_id}")
    if response.status_code == 200:
        print(f"   ✅ 成功获取线程信息")
    else:
        print(f"   ❌ 获取线程信息失败: {response.status_code}")
    
    # 3. 测试/threads/{thread_id}/runs/stream端点
    print(f"\n3. 测试 POST /threads/{thread_id}/runs/stream:")
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "什么是RAG？"
            }
        ],
        "assistant_id": "clip-rag-assistant"
    }
    
    response = requests.post(
        f"{base_url}/threads/{thread_id}/runs/stream",
        json=payload
    )
    
    if response.status_code == 200:
        run_data = response.json()
        run_id = run_data.get("run_id")
        print(f"   ✅ 流式运行创建成功")
        print(f"   运行ID: {run_id}")
        if "messages" in run_data and run_data["messages"]:
            print(f"   回答: {run_data['messages'][0]['content'][:100]}...")
    else:
        print(f"   ❌ 流式运行创建失败: {response.status_code}")
        print(f"   响应: {response.text}")
    
    # 4. 测试/assistants/clip-rag-assistant端点
    print(f"\n4. 测试 GET /assistants/clip-rag-assistant:")
    response = requests.get(f"{base_url}/assistants/clip-rag-assistant")
    if response.status_code == 200:
        print(f"   ✅ 获取助手信息成功")
    else:
        print(f"   ❌ 获取助手信息失败: {response.status_code}")
    
    # 5. 测试/info端点
    print(f"\n5. 测试 GET /info:")
    response = requests.get(f"{base_url}/info")
    if response.status_code == 200:
        info_data = response.json()
        print(f"   ✅ 获取系统信息成功")
        print(f"   可用助手: {info_data.get('assistants')}")
    else:
        print(f"   ❌ 获取系统信息失败: {response.status_code}")
    
    print("\n" + "=" * 60)
    print("测试完成！")

if __name__ == "__main__":
    test_all_endpoints()