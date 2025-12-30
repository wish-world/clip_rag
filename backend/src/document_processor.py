from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from .path_config import get_documents_dir, get_vectorstore_dir
import os
from typing import List
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

class DocumentProcessor:
    def __init__(self, docs_dir: str = None, vectorstore_dir: str = None):
        if docs_dir is None:
            docs_dir = get_documents_dir()
        if vectorstore_dir is None:
            vectorstore_dir = get_vectorstore_dir()
        
        self.docs_dir = str(docs_dir)
        self.vectorstore_dir = str(vectorstore_dir)
        
        print(f"文档处理器初始化:")
        print(f"  文档目录: {self.docs_dir}")
        print(f"  向量存储目录: {self.vectorstore_dir}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
    def load_documents(self) -> List[Document]:
        """加载所有文档"""
        documents = []
        
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir, exist_ok=True)
            print(f"创建了文档目录: {self.docs_dir}")
            return documents
        
        for filename in os.listdir(self.docs_dir):
            file_path = os.path.join(self.docs_dir, filename)
            
            try:
                if filename.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = filename
                        doc.metadata["type"] = "pdf"
                    documents.extend(docs)
                    print(f"✓ 已加载PDF: {filename}, 页数: {len(docs)}")
                elif filename.lower().endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = filename
                        doc.metadata["type"] = "txt"
                    documents.extend(docs)
                    print(f"✓ 已加载TXT: {filename}")
            except Exception as e:
                print(f"✗ 加载文件 {filename} 时出错: {str(e)}")
                continue
                
        return documents
    
    def split_documents(self, documents: List[Document], 
                       chunk_size: int = 1000, 
                       chunk_overlap: int = 200) -> List[Document]:
        """分割文档"""
        if not documents:
            return []
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ";", "；", "，", "、", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        print(f"✓ 文档分割完成: 从 {len(documents)} 个文档分割为 {len(splits)} 个片段")
        return splits
    
    def create_vectorstore(self, documents: List[Document], collection_name: str = "clip_rag"):
        """创建向量存储"""
        if not documents:
            print("✗ 没有文档可处理")
            return None
        
        if not os.path.exists(self.vectorstore_dir):
            os.makedirs(self.vectorstore_dir, exist_ok=True)
        
        print(f"正在创建向量存储，使用 {len(documents)} 个文档片段...")
        
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.vectorstore_dir,
            collection_name=collection_name
        )
        
        vectorstore.persist()
        
        test_results = vectorstore.similarity_search("测试", k=1)
        if test_results:
            print("✓ 向量存储创建成功")
        else:
            print("⚠ 向量存储创建但测试检索无结果")
            
        return vectorstore
    
    def load_or_create_vectorstore(self, collection_name: str = "clip_rag", force_recreate: bool = False):
        """加载或创建向量存储"""
        if force_recreate or not os.path.exists(os.path.join(self.vectorstore_dir, "chroma.sqlite3")):
            print("正在加载文档并创建向量存储...")
            documents = self.load_documents()
            if not documents:
                print(f"✗ 没有找到可处理的文档")
                print(f"请将PDF或TXT文件放入: {self.docs_dir}")
                return None
                
            splits = self.split_documents(documents)
            return self.create_vectorstore(splits, collection_name)
        else:
            print("加载已存在的向量存储...")
            try:
                vectorstore = Chroma(
                    persist_directory=self.vectorstore_dir,
                    embedding_function=self.embeddings,
                    collection_name=collection_name
                )
                test_count = vectorstore._collection.count()
                print(f"✓ 已加载向量存储，包含 {test_count} 个文档片段")
                return vectorstore
            except Exception as e:
                print(f"✗ 加载向量存储失败: {str(e)}")
                print("尝试重新创建...")
                return self.load_or_create_vectorstore(collection_name, force_recreate=True)