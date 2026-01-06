from typing import List, Dict, Any
from langchain_core.documents import Document
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MultimodalFusionStrategy:
    def __init__(self, text_weight: float = 0.7, image_weight: float = 0.3):
        self.text_weight = text_weight
        self.image_weight = image_weight
    
    def fuse_retrieval_results(self, 
                              text_docs: List[Document], 
                              image_docs: List[Document],
                              query: str = "") -> List[Document]:
        """融合文本和图像检索结果"""
        
        if not image_docs:
            return text_docs[:4]  # 只返回文本结果
        
        if not text_docs:
            return image_docs[:4]  # 只返回图像结果
        
        # 为文本和图像结果分配权重
        fused_docs = []
        
        # 文本结果（较高权重）
        for i, doc in enumerate(text_docs[:3]):  # 取前3个文本结果
            fused_doc = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    'fusion_score': 1.0 - (i * 0.1),  # 递减权重
                    'type': 'text',
                    'source_doc': doc
                }
            )
            fused_docs.append(fused_doc)
        
        # 图像结果（较低权重）
        for i, doc in enumerate(image_docs[:2]):  # 取前2个图像结果
            # 为图像文档创建文本描述
            image_description = f"[图像] 来自 {doc.metadata.get('source', '未知')} 第 {doc.metadata.get('page', 1)} 页"
            
            fused_doc = Document(
                page_content=image_description,
                metadata={
                    **doc.metadata,
                    'fusion_score': 0.8 - (i * 0.1),  # 递减权重
                    'type': 'image',
                    'source_doc': doc
                }
            )
            fused_docs.append(fused_doc)
        
        # 按融合分数排序
        fused_docs.sort(key=lambda x: x.metadata.get('fusion_score', 0), reverse=True)
        
        logger.info(f"融合结果: {len(text_docs)}文本 + {len(image_docs)}图像 -> {len(fused_docs)}融合文档")
        
        return fused_docs[:4]  # 返回前4个结果
    
    def is_visual_query(self, query: str) -> bool:
        """判断查询是否与视觉相关"""
        visual_keywords = [
            '图片', '图像', '照片', '图表', '图示', '外观', '颜色', '形状',
            '截图', '照片', '画像', '图形', '可视化', '布局', '设计',
            'picture', 'image', 'photo', 'chart', 'diagram', 'appearance',
            'color', 'shape', 'screenshot', 'visual', 'layout', 'design'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in visual_keywords)
    
    def create_multimodal_context(self, fused_docs: List[Document]) -> str:
        """创建多模态上下文"""
        context_parts = []
        
        for i, doc in enumerate(fused_docs):
            doc_type = doc.metadata.get('type', 'text')
            
            if doc_type == 'text':
                context_parts.append(f"文本片段 {i+1}:\n{doc.page_content}")
            else:  # image
                source = doc.metadata.get('source', '未知文档')
                page = doc.metadata.get('page', 1)
                context_parts.append(f"相关图像 {i+1}: 来自 {source} 第 {page} 页")
        
        return "\n\n".join(context_parts)