"""
拼图数据保存和加载模块
保存为 JSON 格式（包含轨迹、变换矩阵等）
"""

import json
import os
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class MapDataSaver:
    """拼图数据保存器"""
    
    @staticmethod
    def save_map_data(save_dir: str, 
                      canvas: np.ndarray,
                      trajectory: List[Tuple[float, float]],
                      transform_matrix: Optional[np.ndarray] = None,
                      metadata: Optional[Dict] = None,
                      base_name: Optional[str] = None) -> Tuple[str, str]:
        """
        保存拼图数据
        :param save_dir: 保存目录
        :param canvas: 拼图画布
        :param trajectory: 轨迹点列表
        :param transform_matrix: 变换矩阵
        :param metadata: 元数据
        :param base_name: 基础文件名（不含扩展名）
        :return: (图片路径，JSON 路径)
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not base_name:
            base_name = f"map_{timestamp}"
        
        # 1. 保存拼图为图片
        image_path = os.path.join(save_dir, f"{base_name}.jpg")
        cv2.imwrite(image_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # 2. 保存轨迹和元数据为 JSON
        json_path = os.path.join(save_dir, f"{base_name}.json")
        
        data = {
            "version": "1.0",
            "timestamp": timestamp,
            "canvas": {
                "width": canvas.shape[1],
                "height": canvas.shape[0],
                "channels": canvas.shape[2] if len(canvas.shape) > 2 else 1
            },
            "trajectory": {
                "points": [[float(x), float(y)] for x, y in trajectory],
                "point_count": len(trajectory),
                "start_point": [float(trajectory[0][0]), float(trajectory[0][1])] if trajectory else None,
                "end_point": [float(trajectory[-1][0]), float(trajectory[-1][1])] if trajectory else None
            },
            "metadata": metadata or {}
        }
        
        # 添加变换矩阵（如果有）
        if transform_matrix is not None:
            data["transform_matrix"] = transform_matrix.tolist()
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return image_path, json_path
    
    @staticmethod
    def load_map_data(json_path: str) -> Optional[Dict]:
        """
        加载拼图数据
        :param json_path: JSON 文件路径
        :return: 数据字典
        """
        if not os.path.exists(json_path):
            return None
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    @staticmethod
    def load_trajectory_from_json(json_path: str) -> Optional[List[Tuple[float, float]]]:
        """
        从 JSON 文件加载轨迹
        :param json_path: JSON 文件路径
        :return: 轨迹点列表
        """
        data = MapDataSaver.load_map_data(json_path)
        if not data:
            return None
        
        trajectory_data = data.get("trajectory", {})
        points = trajectory_data.get("points", [])
        
        return [(float(x), float(y)) for x, y in points]
    
    @staticmethod
    def list_map_files(save_dir: str) -> List[Dict]:
        """
        列出所有拼图文件
        :param save_dir: 保存目录
        :return: 文件列表
        """
        if not os.path.exists(save_dir):
            return []
        
        files = []
        for filename in os.listdir(save_dir):
            if filename.endswith(".json"):
                json_path = os.path.join(save_dir, filename)
                data = MapDataSaver.load_map_data(json_path)
                
                if data:
                    files.append({
                        "json_path": json_path,
                        "image_path": json_path.replace(".json", ".jpg"),
                        "timestamp": data.get("timestamp", ""),
                        "trajectory_points": data.get("trajectory", {}).get("point_count", 0),
                        "metadata": data.get("metadata", {})
                    })
        
        # 按时间戳排序（最新的在前）
        files.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return files
