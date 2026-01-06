import clip
import torch
import numpy as np
from PIL import Image
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class CLIPProcessor:
    def __init__(self, model_name: str = "ViT-B/32"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self._load_model()
    
    def _load_model(self):
        """加载CLIP模型"""
        try:
            logger.info(f"正在加载CLIP模型: {self.model_name}")
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            logger.info(f"CLIP模型加载成功，设备: {self.device}")
        except Exception as e:
            logger.error(f"CLIP模型加载失败: {e}")
            raise
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """编码单张图像"""
        if self.model is None:
            raise ValueError("CLIP模型未初始化")
        
        try:
            # 预处理图像
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # 生成嵌入向量
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"图像编码失败: {e}")
            raise
    
    def encode_images(self, images: List[Image.Image]) -> List[np.ndarray]:
        """批量编码图像"""
        embeddings = []
        for i, image in enumerate(images):
            try:
                embedding = self.encode_image(image)
                embeddings.append(embedding)
                if (i + 1) % 10 == 0:
                    logger.info(f"已处理 {i + 1}/{len(images)} 张图像")
            except Exception as e:
                logger.error(f"第 {i + 1} 张图像处理失败: {e}")
                continue
                
        return embeddings
    
    def encode_text(self, text: str) -> np.ndarray:
        """编码文本"""
        if self.model is None:
            raise ValueError("CLIP模型未初始化")
        
        try:
            text_tokens = clip.tokenize([text]).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
            return text_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            raise
    
    def get_similarity(self, image_embedding: np.ndarray, text_embedding: np.ndarray) -> float:
        """计算图像和文本的相似度"""
        similarity = np.dot(image_embedding, text_embedding)
        return float(similarity)