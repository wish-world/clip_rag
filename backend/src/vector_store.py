from typing import List, Optional, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field
from .path_config import get_vectorstore_dir
from .multimodal import MultimodalFusionStrategy, CLIPProcessor
import os
import warnings
import logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger(__name__)

class MultiModalRetriever(BaseRetriever):
    """多模态检索器 - 修复版本"""
    
    text_retriever: Any = Field(..., description="文本检索器")
    image_retriever: Optional[Any] = Field(None, description="图像检索器")
    fusion_strategy: Any = Field(..., description="融合策略")
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """检索相关文档（新版本API）"""
        # 文本检索
        text_docs = []
        if self.text_retriever:
            try:
                # 使用新版本的invoke方法
                if hasattr(self.text_retriever, 'invoke'):
                    text_docs = self.text_retriever.invoke(query)
                else:
                    # 回退到旧版本方法
                    text_docs = self.text_retriever.get_relevant_documents(query)
            except Exception as e:
                logger.error(f"文本检索失败: {e}")
        
        # 图像检索（如果是视觉查询）
        image_docs = []
        if self.image_retriever and hasattr(self.fusion_strategy, 'is_visual_query') and self.fusion_strategy.is_visual_query(query):
            try:
                if hasattr(self.image_retriever, 'invoke'):
                    image_docs = self.image_retriever.invoke(query)
                else:
                    image_docs = self.image_retriever.get_relevant_documents(query)
            except Exception as e:
                logger.warning(f"图像检索失败: {e}")
        
        # 融合结果
        if hasattr(self.fusion_strategy, 'fuse_retrieval_results'):
            fused_docs = self.fusion_strategy.fuse_retrieval_results(text_docs, image_docs, query)
        else:
            # 简单的融合策略
            fused_docs = text_docs + image_docs
        
        logger.info(f"多模态检索完成: {len(text_docs)}文本 + {len(image_docs)}图像 -> {len(fused_docs)}结果")
        return fused_docs
    
    # 兼容旧版本的方法
    def get_relevant_documents(self, query: str) -> List[Document]:
        """兼容旧版本API"""
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        return self._get_relevant_documents(query, run_manager=None)
    
    async def _aget_relevant_documents(
        self, 
        query: str, 
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """异步检索相关文档"""
        return self._get_relevant_documents(query, run_manager=run_manager)
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """异步检索相关文档"""
        return await self._aget_relevant_documents(query, run_manager=None)

class MultiModalVectorStoreManager:
    def __init__(self, vectorstore_dir: str = None):
        if vectorstore_dir is None:
            vectorstore_dir = get_vectorstore_dir()
        
        self.vectorstore_dir = str(vectorstore_dir)
        
        print(f"多模态向量存储管理器初始化:")
        print(f"  向量存储目录: {self.vectorstore_dir}")
        
        # 文本嵌入模型
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # 向量存储实例
        self.text_vectorstore: Optional[Chroma] = None
        self.image_vectorstore: Optional[Chroma] = None
        
        # 多模态组件
        self.clip_processor = None
        self.fusion_strategy = MultimodalFusionStrategy()
        
        self._init_multimodal_components()
    
    def _init_multimodal_components(self):
        """初始化多模态组件"""
        try:
            from .multimodal import CLIPProcessor
            self.clip_processor = CLIPProcessor()
            print("✓ CLIP处理器初始化成功")
        except Exception as e:
            print(f"⚠ CLIP处理器初始化失败: {e}")
            print("⚠ 多模态功能将不可用，仅支持文本处理")
    
    def initialize_vectorstore(self, collection_name: str = "clip_rag") -> bool:
        """初始化向量存储"""
        chroma_db_file = os.path.join(self.vectorstore_dir, "chroma.sqlite3")
        
        if not os.path.exists(chroma_db_file):
            print(f"✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗ 向量存储文件不存在: {chroma_db_file}")
            print(f"请先运行 python initialize_vectorstore.py 初始化向量存储")
            return False
        
        print(f"加载已存在的多模态向量存储: {self.vectorstore_dir}")
        
        success = True
        
        # 加载文本向量存储
        try:
            self.text_vectorstore = Chroma(
                persist_directory=self.vectorstore_dir,
                embedding_function=self.text_embeddings,
                collection_name=f"{collection_name}_text"
            )
            text_count = self.text_vectorstore._collection.count()
            print(f"✓ 已加载文本向量存储，包含 {text_count} 个文档片段")
        except Exception as e:
            print(f"✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗ 加载文本向量存储失败: {str(e)}")
            self.text_vectorstore = None
            success = False
        
        # 加载图像向量存储（如果CLIP可用）
        if self.clip_processor:
            try:
                from langchain_core.embeddings import Embeddings
                
                class CLIPEmbeddings(Embeddings):
                    def __init__(self, clip_processor):
                        self.clip_processor = clip_processor
                    
                    def embed_documents(self, texts):
                        return [self.clip_processor.encode_text(text).tolist() for text in texts]
                    
                    def embed_query(self, text):
                        return self.clip_processor.encode_text(text).tolist()
                
                clip_embeddings = CLIPEmbeddings(self.clip_processor)
                
                self.image_vectorstore = Chroma(
                    persist_directory=self.vectorstore_dir,
                    embedding_function=clip_embeddings,
                    collection_name=f"{collection_name}_images"
                )
                image_count = self.image_vectorstore._collection.count()
                print(f"✓ 已加载图像向量存储，包含 {image_count} 个图像片段")
            except Exception as e:
                print(f"⚠ 加载图像向量存储失败: {e}")
                self.image_vectorstore = None
        else:
            print("⚠ CLIP不可用，跳过图像向量存储加载")
        
        return success
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """多模态相似性搜索"""
        text_docs = []
        image_docs = []
        
        # 文本检索
        if self.text_vectorstore:
            try:
                text_docs = self.text_vectorstore.similarity_search(query, k=k)
            except Exception as e:
                logger.error(f"文本检索失败: {e}")
        
        # 图像检索（如果是视觉查询）
        if self.image_vectorstore and hasattr(self.fusion_strategy, 'is_visual_query') and self.fusion_strategy.is_visual_query(query):
            try:
                image_docs = self.image_vectorstore.similarity_search(query, k=min(2, k))
            except Exception as e:
                logger.warning(f"图像检索失败: {e}")
        
        # 融合结果
        if hasattr(self.fusion_strategy, 'fuse_retrieval_results'):
            fused_docs = self.fusion_strategy.fuse_retrieval_results(text_docs, image_docs, query)
        else:
            fused_docs = text_docs + image_docs
        
        return fused_docs[:k]
    
    def as_retriever(self, search_kwargs: dict = None):
        """获取多模态检索器"""
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        # 创建文本检索器
        text_retriever = None
        if self.text_vectorstore:
            try:
                text_retriever = self.text_vectorstore.as_retriever(search_kwargs=search_kwargs)
            except Exception as e:
                logger.error(f"创建文本检索器失败: {e}")
        
        # 创建图像检索器
        image_retriever = None
        if self.image_vectorstore:
            try:
                image_search_kwargs = search_kwargs.copy()
                image_search_kwargs["k"] = min(2, search_kwargs.get("k", 4))
                image_retriever = self.image_vectorstore.as_retriever(search_kwargs=image_search_kwargs)
            except Exception as e:
                logger.error(f"创建图像检索器失败: {e}")
        
        # 如果只有文本检索器，直接返回
        if not image_retriever:
            if text_retriever:
                return text_retriever
            else:
                raise ValueError("没有可用的向量存储")
        
        # 返回多模态检索器
        return MultiModalRetriever(
            text_retriever=text_retriever,
            image_retriever=image_retriever,
            fusion_strategy=self.fusion_strategy
        )
    
    def get_text_retriever(self, search_kwargs: dict = None):
        """获取纯文本检索器"""
        if not self.text_vectorstore:
            raise ValueError("文本向量存储未初始化")
        
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        try:
            return self.text_vectorstore.as_retriever(search_kwargs=search_kwargs)
        except Exception as e:
            logger.error(f"获取文本检索器失败: {e}")
            # 回退到直接搜索
            def simple_retriever(query: str) -> List[Document]:
                return self.text_vectorstore.similarity_search(query, k=search_kwargs.get("k", 4))
            return simple_retriever
    
    def get_status(self) -> Dict[str, Any]:
        """获取向量存储状态"""
        text_count = 0
        image_count = 0
        
        if self.text_vectorstore:
            try:
                text_count = self.text_vectorstore._collection.count()
            except:
                pass
        
        if self.image_vectorstore:
            try:
                image_count = self.image_vectorstore._collection.count()
            except:
                pass
        
        return {
            "text_initialized": self.text_vectorstore is not None,
            "image_initialized": self.image_vectorstore is not None,
            "text_document_count": text_count,
            "image_document_count": image_count,
            "multimodal_enabled": self.clip_processor is not None,
            "vectorstore_path": self.vectorstore_dir
        }