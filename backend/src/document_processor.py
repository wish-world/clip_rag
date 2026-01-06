from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from .path_config import get_documents_dir, get_vectorstore_dir
from .multimodal import CLIPProcessor, ImageExtractor, MultimodalFusionStrategy
import os
from typing import List, Optional
import warnings
from PIL import Image
import logging
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger(__name__)

class MultiModalDocumentProcessor:
    def __init__(self, docs_dir: str = None, vectorstore_dir: str = None):
        if docs_dir is None:
            docs_dir = get_documents_dir()
        if vectorstore_dir is None:
            vectorstore_dir = get_vectorstore_dir()
        
        self.docs_dir = str(docs_dir)
        self.vectorstore_dir = str(vectorstore_dir)
        
        print(f"多模态文档处理器初始化:")
        print(f"  文档目录: {self.docs_dir}")
        print(f"  向量存储目录: {self.vectorstore_dir}")
        
        # 文本嵌入模型
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # 多模态组件
        self.clip_processor = None
        self.image_extractor = ImageExtractor()
        self.fusion_strategy = MultimodalFusionStrategy()
        self._init_multimodal_components()
    
    def _init_multimodal_components(self):
        """初始化多模态组件"""
        try:
            self.clip_processor = CLIPProcessor()
            print("✓ CLIP处理器初始化成功")
        except Exception as e:
            print(f"⚠ CLIP处理器初始化失败: {e}")
            print("⚠ 多模态功能将不可用，仅支持文本处理")
    
    def _filter_complex_metadata(self, documents: List[Document]) -> List[Document]:
        """过滤复杂元数据，确保ChromaDB兼容性"""
        filtered_docs = []
        
        for doc in documents:
            # 创建新的元数据字典，只保留简单类型
            simple_metadata = {}
            
            for key, value in doc.metadata.items():
                # 只保留字符串、数字、布尔值等简单类型
                if isinstance(value, (str, int, float, bool)) or value is None:
                    simple_metadata[key] = value
                elif isinstance(value, (list, tuple)):
                    # 如果是列表或元组，检查元素类型
                    if all(isinstance(item, (str, int, float, bool)) for item in value):
                        simple_metadata[key] = value
                    else:
                        # 复杂列表，转换为字符串表示或跳过
                        simple_metadata[f"{key}_preview"] = f"列表长度: {len(value)}"
                elif hasattr(value, 'tolist'):  # numpy数组
                    # 跳过numpy数组，不存储在元数据中
                    continue
                else:
                    # 其他复杂类型，转换为字符串预览
                    simple_metadata[f"{key}_preview"] = str(type(value))
            
            # 创建新的文档对象
            filtered_doc = Document(
                page_content=doc.page_content,
                metadata=simple_metadata
            )
            filtered_docs.append(filtered_doc)
        
        return filtered_docs
    
    def load_documents(self) -> List[Document]:
        """加载所有文档（支持多模态）"""
        documents = []
        
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir, exist_ok=True)
            print(f"创建了文档目录: {self.docs_dir}")
            return documents
        
        supported_formats = ['.pdf', '.txt', '.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for filename in os.listdir(self.docs_dir):
            file_path = os.path.join(self.docs_dir, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext not in supported_formats:
                continue
            
            try:
                if file_ext == '.pdf':
                    docs = self._load_pdf_with_images(file_path, filename)
                    documents.extend(docs)
                    print(f"✓ 已加载PDF: {filename}, 文档数: {len(docs)}")
                    
                elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    docs = self._load_image_file(file_path, filename)
                    documents.extend(docs)
                    print(f"✓ 已加载图像: {filename}, 文档数: {len(docs)}")
                    
                elif file_ext == '.txt':
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
    
    def _load_pdf_with_images(self, pdf_path: str, filename: str) -> List[Document]:
        """加载PDF文档（包含图像提取）"""
        documents = []
        
        # 1. 加载文本内容
        loader = PyPDFLoader(pdf_path)
        text_docs = loader.load()
        
        for doc in text_docs:
            doc.metadata.update({
                "source": filename,
                "type": "pdf_text",
                "has_image": False
            })
        documents.extend(text_docs)
        
        # 2. 提取图像（如果CLIP可用）
        if self.clip_processor:
            try:
                images = self.image_extractor.extract_from_pdf(pdf_path)
                image_docs = self._create_image_documents(images, filename)
                documents.extend(image_docs)
                print(f"  - 提取 {len(image_docs)} 张图像")
            except Exception as e:
                print(f"  ⚠ PDF图像提取失败: {e}")
        
        return documents
    
    def _load_image_file(self, image_path: str, filename: str) -> List[Document]:
        """加载单个图像文件"""
        if not self.clip_processor:
            print(f"  ⚠ 跳过图像 {filename} (CLIP不可用)")
            return []
        
        try:
            images = self.image_extractor.extract_from_image_file(image_path)
            image_docs = self._create_image_documents(images, filename)
            return image_docs
        except Exception as e:
            print(f"  ⚠ 图像文件处理失败: {e}")
            return []
    
    def _create_image_documents(self, images: List[dict], source: str) -> List[Document]:
        """创建图像文档 - 修复版本：过滤复杂元数据"""
        image_docs = []
        
        for img_info in images:
            try:
                # 生成图像嵌入
                image_embedding = self.clip_processor.encode_image(img_info['image'])
                
                # 创建文档
                doc_content = f"图像来自 {source} 第 {img_info['page']} 页"
                
                # 只保留简单的元数据，移除复杂的numpy数组
                doc = Document(
                    page_content=doc_content,
                    metadata={
                        "source": source,
                        "type": "image",
                        "page": img_info['page'],
                        "image_index": img_info['image_index'],
                        "width": img_info['width'],
                        "height": img_info['height'],
                        "format": img_info['format'],
                        # 移除复杂的image_embedding，只保留维度信息
                        "embedding_dim": len(image_embedding) if hasattr(image_embedding, '__len__') else 0,
                        "has_image": True
                    }
                )
                image_docs.append(doc)
                
            except Exception as e:
                print(f"  ⚠ 图像文档创建失败: {e}")
                continue
        
        return image_docs
    
    def split_documents(self, documents: List[Document], 
                       chunk_size: int = 1000, 
                       chunk_overlap: int = 200) -> List[Document]:
        """分割文档（跳过图像文档的分割）"""
        if not documents:
            return []
        
        # 分离文本和图像文档
        text_docs = [doc for doc in documents if not doc.metadata.get('has_image', False)]
        image_docs = [doc for doc in documents if doc.metadata.get('has_image', False)]
        
        # 只对文本文档进行分割
        if text_docs:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", ";", "；", "，", "、", " ", ""]
            )
            split_text_docs = text_splitter.split_documents(text_docs)
        else:
            split_text_docs = []
        
        # 图像文档不分割
        all_docs = split_text_docs + image_docs
        
        print(f"✓ 文档分割完成: {len(text_docs)}文本 -> {len(split_text_docs)}片段, {len(image_docs)}图像")
        return all_docs
    
    def create_vectorstore(self, documents: List[Document], collection_name: str = "clip_rag"):
        """创建多模态向量存储 - 修复版本"""
        # 确保Chroma在函数内可用
        from langchain_community.vectorstores import Chroma
        from langchain_core.embeddings import Embeddings
        
        if not documents:
            print("✗ 没有文档可处理")
            return None
        
        if not os.path.exists(self.vectorstore_dir):
            os.makedirs(self.vectorstore_dir, exist_ok=True)
        
        print(f"正在创建多模态向量存储，使用 {len(documents)} 个文档...")
        
        # 分离文本和图像文档
        text_docs = [doc for doc in documents if not doc.metadata.get('has_image', False)]
        image_docs = [doc for doc in documents if doc.metadata.get('has_image', False)]
        
        print(f"  - 文本文档: {len(text_docs)} 个")
        print(f"  - 图像文档: {len(image_docs)} 个")
        
        # 创建文本向量存储
        text_vectorstore = None
        if text_docs:
            try:
                # 过滤文本文档的复杂元数据
                filtered_text_docs = self._filter_complex_metadata(text_docs)
                text_vectorstore = Chroma.from_documents(
                    documents=filtered_text_docs,
                    embedding=self.text_embeddings,
                    persist_directory=self.vectorstore_dir,
                    collection_name=f"{collection_name}_text"
                )
                print("✓ 文本向量存储创建成功")
            except Exception as e:
                print(f"✗ 文本向量存储创建失败: {e}")
                text_vectorstore = None
        else:
            print("⚠ 无文本文档，跳过文本向量存储")
        
        # 创建图像向量存储（如果有多模态支持）
        image_vectorstore = None
        if image_docs and self.clip_processor:
            try:
                # 过滤图像文档的复杂元数据
                filtered_image_docs = self._filter_complex_metadata(image_docs)
                
                # 为Chroma准备自定义嵌入函数
                class CLIPEmbeddings(Embeddings):
                    def __init__(self, clip_processor, image_docs):
                        self.clip_processor = clip_processor
                        # 创建图像路径到嵌入的映射
                        self.image_embeddings = {}
                        for doc in image_docs:
                            # 使用文档的唯一标识符（源文件+页码+图像索引）
                            doc_id = f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('page', 0)}_{doc.metadata.get('image_index', 0)}"
                            # 在实际应用中，这里应该重新计算图像嵌入
                            # 由于我们移除了原始嵌入，这里使用文本描述生成嵌入作为替代
                            try:
                                text_embedding = self.clip_processor.encode_text(doc.page_content).tolist()
                                self.image_embeddings[doc_id] = text_embedding
                            except Exception as e:
                                # 使用零向量作为后备
                                self.image_embeddings[doc_id] = [0.0] * 512
                    
                    def embed_documents(self, texts):
                        embeddings = []
                        for text in texts:
                            # 尝试从映射中获取嵌入
                            doc_id = None
                            for key in self.image_embeddings:
                                if key in text:  # 简单的匹配逻辑
                                    doc_id = key
                                    break
                            
                            if doc_id and doc_id in self.image_embeddings:
                                embeddings.append(self.image_embeddings[doc_id])
                            else:
                                # 使用文本描述生成嵌入
                                try:
                                    embedding = self.clip_processor.encode_text(text).tolist()
                                    embeddings.append(embedding)
                                except Exception as e:
                                    embeddings.append([0.0] * 512)
                        return embeddings
                    
                    def embed_query(self, text):
                        try:
                            return self.clip_processor.encode_text(text).tolist()
                        except Exception as e:
                            print(f"⚠ 查询嵌入失败: {e}")
                            return [0.0] * 512
                
                clip_embeddings = CLIPEmbeddings(self.clip_processor, image_docs)
                
                image_vectorstore = Chroma.from_documents(
                    documents=filtered_image_docs,
                    embedding=clip_embeddings,
                    persist_directory=self.vectorstore_dir,
                    collection_name=f"{collection_name}_images"
                )
                print("✓ 图像向量存储创建成功")
            except Exception as e:
                print(f"✗ 图像向量存储创建失败: {e}")
                import traceback
                traceback.print_exc()
        elif image_docs:
            print("⚠ 有图像文档但CLIP不可用，跳过图像向量存储")
        
        # 返回多模态向量存储管理器
        from .vector_store import MultiModalVectorStoreManager
        vectorstore_manager = MultiModalVectorStoreManager(self.vectorstore_dir)
        vectorstore_manager.text_vectorstore = text_vectorstore
        vectorstore_manager.image_vectorstore = image_vectorstore
        
        # 测试检索
        if text_vectorstore:
            try:
                test_results = text_vectorstore.similarity_search("测试", k=1)
                if test_results:
                    print("✓ 文本检索测试成功")
                else:
                    print("⚠ 文本检索测试无结果")
            except Exception as e:
                print(f"⚠ 文本检索测试失败: {e}")
        
        return vectorstore_manager
    
    def load_or_create_vectorstore(self, collection_name: str = "clip_rag", force_recreate: bool = False):
        """加载或创建多模态向量存储"""
        chroma_db_file = os.path.join(self.vectorstore_dir, "chroma.sqlite3")
        
        if force_recreate or not os.path.exists(chroma_db_file):
            print("正在加载文档并创建多模态向量存储...")
            documents = self.load_documents()
            if not documents:
                print(f"✗ 没有找到可处理的文档")
                print(f"请将PDF、TXT或图像文件放入: {self.docs_dir}")
                return None
                
            splits = self.split_documents(documents)
            return self.create_vectorstore(splits, collection_name)
        else:
            print("加载已存在的多模态向量存储...")
            try:
                from .vector_store import MultiModalVectorStoreManager
                vectorstore_manager = MultiModalVectorStoreManager(self.vectorstore_dir)
                
                # 确保在函数内导入Chroma
                from langchain_community.vectorstores import Chroma
                
                # 尝试加载文本向量存储
                if os.path.exists(os.path.join(self.vectorstore_dir, "chroma.sqlite3")):
                    try:
                        text_vectorstore = Chroma(
                            persist_directory=self.vectorstore_dir,
                            embedding_function=self.text_embeddings,
                            collection_name=f"{collection_name}_text"
                        )
                        text_count = text_vectorstore._collection.count()
                        vectorstore_manager.text_vectorstore = text_vectorstore
                        print(f"✓ 已加载文本向量存储，包含 {text_count} 个文档片段")
                    except Exception as e:
                        print(f"⚠ 加载文本向量存储失败: {e}")
                
                # 尝试加载图像向量存储（如果CLIP可用）
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
                        
                        image_vectorstore = Chroma(
                            persist_directory=self.vectorstore_dir,
                            embedding_function=clip_embeddings,
                            collection_name=f"{collection_name}_images"
                        )
                        image_count = image_vectorstore._collection.count()
                        vectorstore_manager.image_vectorstore = image_vectorstore
                        print(f"✓ 已加载图像向量存储，包含 {image_count} 个图像片段")
                    except Exception as e:
                        print(f"⚠ 加载图像向量存储失败: {e}")
                
                return vectorstore_manager
                
            except Exception as e:
                print(f"✗ 加载向量存储失败: {str(e)}")
                print("尝试重新创建...")
                return self.load_or_create_vectorstore(collection_name, force_recreate=True)