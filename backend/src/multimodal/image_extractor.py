import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageOps
from typing import List, Dict, Any
import io
import os
import logging

logger = logging.getLogger(__name__)

class ImageExtractor:
    def __init__(self, max_image_size: int = 1024):
        self.max_image_size = max_image_size
    
    def extract_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """从PDF中提取图像"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        images = []
        try:
            pdf_document = fitz.open(pdf_path)
            logger.info(f"正在处理PDF: {os.path.basename(pdf_path)}, 共 {len(pdf_document)} 页")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images()
                
                logger.debug(f"第 {page_num + 1} 页找到 {len(image_list)} 张图像")
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_data = base_image["image"]
                        
                        # 转换为PIL Image
                        image = Image.open(io.BytesIO(image_data))
                        
                        # 调整图像大小
                        image = self._resize_image(image)
                        
                        image_info = {
                            'image': image,
                            'page': page_num + 1,
                            'image_index': img_index,
                            'source': os.path.basename(pdf_path),
                            'width': image.width,
                            'height': image.height,
                            'format': base_image.get("ext", "unknown")
                        }
                        
                        images.append(image_info)
                        
                    except Exception as e:
                        logger.warning(f"提取第 {page_num + 1} 页第 {img_index + 1} 张图像失败: {e}")
                        continue
            
            pdf_document.close()
            logger.info(f"成功提取 {len(images)} 张图像")
            
        except Exception as e:
            logger.error(f"PDF处理失败: {e}")
            raise
        
        return images
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """调整图像大小"""
        if max(image.size) <= self.max_image_size:
            return image
        
        # 保持宽高比调整大小
        image.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)
        return image
    
    def extract_from_image_file(self, image_path: str) -> List[Dict[str, Any]]:
        """从图像文件中提取"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        try:
            image = Image.open(image_path)
            image = self._resize_image(image)
            
            image_info = {
                'image': image,
                'page': 1,
                'image_index': 0,
                'source': os.path.basename(image_path),
                'width': image.width,
                'height': image.height,
                'format': image.format or "unknown"
            }
            
            return [image_info]
            
        except Exception as e:
            logger.error(f"图像文件处理失败: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """图像预处理"""
        # 转换为RGB（如果必要）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 自动旋转（基于EXIF）
        image = ImageOps.exif_transpose(image)
        
        return image