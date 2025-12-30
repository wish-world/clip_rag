from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from .path_config import get_vectorstore_dir
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

class VectorStoreManager:
    def __init__(self, vectorstore_dir: str = None):
        if vectorstore_dir is None:
            vectorstore_dir = get_vectorstore_dir()
        
        self.vectorstore_dir = str(vectorstore_dir)
        
        print(f"向量存储管理器初始化:")
        print(f"  向量存储目录: {self.vectorstore_dir}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vectorstore: Optional[Chroma] = None
        
    def initialize_vectorstore(self, collection_name: str = "clip_rag"):
        """初始化向量存储"""
        chroma_db_file = os.path.join(self.vectorstore_dir, "chroma.sqlite3")
        
        if not os.path.exists(chroma_db_file):
            print(f"✗ 向量存储文件不存在: {chroma_db_file}")
            print(f"请先运行 python initialize_vectorstore.py 初始化向量存储")
            return False
        
        print(f"加载已存在的向量存储: {self.vectorstore_dir}")
        try:
            self.vectorstore = Chroma(
                persist_directory=self.vectorstore_dir,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            count = self.vectorstore._collection.count()
            print(f"✓ 已加载向量存储，包含 {count} 个文档片段")
            return True
        except Exception as e:
            print(f"✗ 加载向量存储失败: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """相似性搜索"""
        if not self.vectorstore:
            raise ValueError("向量存储未初始化")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def as_retriever(self, search_kwargs: dict = None):
        """获取检索器"""
        if not self.vectorstore:
            raise ValueError("向量存储未初始化")
        
        if search_kwargs is None:
            search_kwargs = {"k": 4}
            
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)