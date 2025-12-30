import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.document_processor import DocumentProcessor
    from src.path_config import get_documents_dir, get_vectorstore_dir
    print("✓ 成功导入模块")
except ImportError as e:
    print(f"✗ 导入错误: {e}")
    sys.exit(1)

def main():
    print("=" * 60)
    print("CLIP RAG 向量存储初始化")
    print("=" * 60)
    
    docs_dir = get_documents_dir()
    vectorstore_dir = get_vectorstore_dir()
    
    print(f"文档目录: {docs_dir}")
    print(f"向量存储目录: {vectorstore_dir}")
    
    docs_dir.mkdir(exist_ok=True)
    vectorstore_dir.mkdir(exist_ok=True)
    
    print(f"\n检查文档目录: {docs_dir}")
    files = [f for f in os.listdir(docs_dir) if f.lower().endswith(('.pdf', '.txt'))]
    if files:
        print(f"找到 {len(files)} 个文档:")
        for f in files[:10]:
            print(f"  - {f}")
        if len(files) > 10:
            print(f"  ... 和 {len(files) - 10} 个其他文件")
    else:
        print("⚠ 没有找到PDF或TXT文档")
        print(f"请将文档放入: {docs_dir}")
        return
    
    processor = DocumentProcessor()
    
    print("\n开始创建向量存储...")
    vectorstore = processor.load_or_create_vectorstore()
    
    if vectorstore:
        print("\n" + "=" * 60)
        print("✅ 向量存储初始化完成！")
        print(f"位置: {vectorstore_dir}")
        
        print("\n测试检索功能...")
        try:
            test_results = vectorstore.similarity_search("测试", k=2)
            if test_results:
                print(f"检索测试成功，找到 {len(test_results)} 个结果")
                for i, result in enumerate(test_results[:2]):
                    print(f"示例片段 {i+1}: {result.page_content[:100]}...")
            else:
                print("⚠ 检索测试无结果，但向量存储已创建")
        except Exception as e:
            print(f"⚠ 检索测试出错: {e}")
            
        print("\n下一步:")
        print("1. 启动后端服务器: python start_server.py")
        print("2. 进入前端目录: cd ../frontend")
        print("3. 安装前端依赖: pnpm install")
        print("4. 启动前端: pnpm dev")
        print("5. 在浏览器中访问: http://localhost:3000")
    else:
        print("\n❌ 向量存储创建失败")
        print("请检查文档和依赖是否正确安装")

if __name__ == "__main__":
    main()