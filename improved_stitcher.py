import numpy as np
import cv2
from typing import Optional, Tuple, Dict, List
import logging
import pickle

logger = logging.getLogger(__name__)


class ImprovedMiniMapStitcher:
    """改进版小地图拼接类 - 专为网页拼接优化"""

    def __init__(self):
        # 使用 ORB 算法（更适合文字内容）
        self.detector = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
        # 使用更精确的匹配器
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=100)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        self.stitched_map: Optional[np.ndarray] = None
        self.current_offset: np.ndarray = np.array([0, 0], dtype=np.float32)
        self.map_data: Dict = {
            'images': [],
            'positions': []
        }
        self.last_frame: Optional[np.ndarray] = None
        self.last_transform: Optional[np.ndarray] = None
        self.accumulated_transform: np.ndarray = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        
        # 历史匹配记录，用于回滚
        self.transform_history: List[Dict] = []
        self.frame_history: List[np.ndarray] = []

    def add_frame(self, frame: np.ndarray) -> bool:
        """
        添加一帧图像到拼接地图

        Args:
            frame: 输入的小地图图像

        Returns:
            是否成功添加
        """
        if frame is None or frame.size == 0:
            return False

        try:
            # 确保图像是 3 通道 RGB
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

            if self.stitched_map is None:
                # 第一帧，初始化
                self._init_map(frame)
                return True
            else:
                # 与上一帧匹配
                success, transform = self._match_frames(self.last_frame, frame)

                if success:
                    self._update_map(frame, transform)
                    return True
                else:
                    logger.warning("匹配失败，尝试与更早的帧匹配")
                    # 匹配失败，尝试与历史帧匹配
                    success = self._fallback_match(frame)
                    return success

        except Exception as e:
            logger.error(f"添加帧失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _init_map(self, frame: np.ndarray):
        """初始化地图"""
        self.stitched_map = frame.copy()
        self.current_offset = np.array([frame.shape[1] // 2, frame.shape[0] // 2], dtype=np.float32)
        self._save_frame_data(frame, np.array([0, 0]))
        self.last_frame = frame.copy()
        self.frame_history.append(frame.copy())

    def _match_frames(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """匹配两帧图像，计算变换矩阵（改进版）"""
        try:
            # 检测特征点和描述符
            kp1, des1 = self.detector.detectAndCompute(img1, None)
            kp2, des2 = self.detector.detectAndCompute(img2, None)

            if des1 is None or des2 is None or len(des1) < 20 or len(des2) < 20:
                logger.warning(f"特征点不足: img1={len(des1) if des1 is not None else 0}, img2={len(des2) if des2 is not None else 0}")
                return False, None

            # 特征匹配
            matches = self.matcher.knnMatch(des1, des2, k=2)

            # 改进的 Lowe's ratio test
            good_matches = []
            for m_n in matches:
                if len(m_n) < 2:
                    continue
                m, n = m_n
                if m.distance < 0.8 * n.distance:  # 放宽一点阈值
                    good_matches.append(m)

            if len(good_matches) < 15:
                logger.warning(f"好的匹配点不足: {len(good_matches)}")
                return False, None

            # 提取匹配点
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 计算变换矩阵（使用 RANSAC 去除异常值）
            M, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC, 
                                                 ransacReprojThreshold=5.0, maxIters=2000)

            if M is not None:
                # 验证变换矩阵是否合理
                if self._is_transform_valid(M):
                    return True, M
                else:
                    logger.warning("变换矩阵不合理，拒绝匹配")
                    return False, None
            else:
                return False, None

        except Exception as e:
            logger.error(f"帧匹配失败: {e}")
            import traceback
            traceback.print_exc()
            return False, None

    def _is_transform_valid(self, M: np.ndarray) -> bool:
        """验证变换矩阵是否合理"""
        # 检查是否旋转太多（网页一般不会旋转）
        angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
        if abs(angle) > 30:
            logger.warning(f"旋转角度过大: {angle:.1f}°")
            return False
        
        # 检查是否缩放太多
        scale_x = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
        scale_y = np.sqrt(M[1, 0]**2 + M[1, 1]**2)
        if scale_x < 0.5 or scale_x > 2.0 or scale_y < 0.5 or scale_y > 2.0:
            logger.warning(f"缩放比例不合理: x={scale_x:.2f}, y={scale_y:.2f}")
            return False
        
        # 检查是否平移太多
        dx, dy = M[0, 2], M[1, 2]
        if abs(dx) > 500 or abs(dy) > 500:
            logger.warning(f"平移距离过大: dx={dx:.0f}, dy={dy:.0f}")
            return False
        
        return True

    def _fallback_match(self, frame: np.ndarray) -> bool:
        """回退匹配：尝试与历史帧匹配"""
        # 最多尝试匹配前 5 帧
        for i in range(min(len(self.frame_history), 5)):
            if i == 0:
                continue  # last_frame 已经试过了
            
            past_frame = self.frame_history[-(i+1)]
            success, transform = self._match_frames(past_frame, frame)
            if success:
                logger.info(f"回退匹配成功: 与前 {i+1} 帧匹配")
                
                # 补偿累积变换
                if len(self.transform_history) >= i:
                    for j in range(len(self.transform_history) - i, len(self.transform_history)):
                        transform = self._compose_transforms(self.transform_history[j]['transform'], transform)
                
                self._update_map(frame, transform)
                return True
        
        return False

    def _compose_transforms(self, M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
        """组合两个变换矩阵"""
        M1_homogeneous = np.vstack([M1, [0, 0, 1]])
        M2_homogeneous = np.vstack([M2, [0, 0, 1]])
        M_combined = M1_homogeneous @ M2_homogeneous
        return M_combined[:2, :]

    def _update_map(self, new_frame: np.ndarray, transform: np.ndarray):
        """更新拼接地图（改进版）"""
        try:
            # 更新偏移
            dx = transform[0, 2]
            dy = transform[1, 2]
            self.current_offset += np.array([dx, dy], dtype=np.float32)

            # 保存当前帧数据
            self._save_frame_data(new_frame, self.current_offset.copy())
            self.frame_history.append(new_frame.copy())
            self.transform_history.append({
                'transform': transform.copy(),
                'offset': self.current_offset.copy()
            })

            # 获取新帧的大小
            h, w = new_frame.shape[:2]

            # 计算新帧在拼接图中的位置
            new_center = self.current_offset
            new_x1 = int(new_center[0] - w // 2)
            new_y1 = int(new_center[1] - h // 2)
            new_x2 = new_x1 + w
            new_y2 = new_y1 + h

            # 检查是否需要扩展地图
            pad_top = max(0, -new_y1)
            pad_bottom = max(0, new_y2 - self.stitched_map.shape[0])
            pad_left = max(0, -new_x1)
            pad_right = max(0, new_x2 - self.stitched_map.shape[1])

            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                self.stitched_map = cv2.copyMakeBorder(
                    self.stitched_map,
                    pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0)
                )
                # 更新偏移以反映填充
                self.current_offset += np.array([pad_left, pad_top], dtype=np.float32)
                new_x1 += pad_left
                new_y1 += pad_top

            # 计算粘贴区域
            h_map, w_map = self.stitched_map.shape[:2]
            paste_x1 = max(0, new_x1)
            paste_y1 = max(0, new_y1)
            paste_x2 = min(w_map, new_x1 + w)
            paste_y2 = min(h_map, new_y1 + h)

            # 计算新帧的裁剪区域
            crop_x1 = paste_x1 - new_x1
            crop_y1 = paste_y1 - new_y1
            crop_x2 = crop_x1 + (paste_x2 - paste_x1)
            crop_y2 = crop_y1 + (paste_y2 - paste_y1)

            # 混合新帧到地图（改进的混合策略）
            if (paste_x2 > paste_x1 and paste_y2 > paste_y1 and
                    crop_x2 > crop_x1 and crop_y2 > crop_y1):
                new_region = new_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                map_region = self.stitched_map[paste_y1:paste_y2, paste_x1:paste_x2]
                
                self._blend_images(map_region, new_region)

            self.last_frame = new_frame.copy()

        except Exception as e:
            logger.error(f"更新地图失败: {e}")
            import traceback
            traceback.print_exc()

    def _blend_images(self, base_region: np.ndarray, new_region: np.ndarray):
        """改进的图像混合策略"""
        # 判断 base_region 是否为初始化的黑色背景
        base_is_black = np.mean(base_region) < 5
        
        if base_is_black:
            # 如果背景是黑的，直接覆盖
            base_region[:] = new_region
        else:
            # 计算两个区域的亮度
            base_luminance = np.mean(base_region, axis=2, keepdims=True)
            new_luminance = np.mean(new_region, axis=2, keepdims=True)
            
            # 根据亮度决定如何混合
            # 策略1: 新区域更亮 → 优先用新区域
            mask_bright = new_luminance > base_luminance * 1.1
            
            # 策略2: 旧区域有内容但新区域不同 → 保留两者的较好部分
            mask_old_valid = base_luminance > 10
            
            # 策略3: 重叠区域用加权平均
            overlap_mask = mask_old_valid & (np.abs(new_luminance - base_luminance) < 30)
            
            # 应用混合
            # 情况1: 新区域更亮 → 替换
            base_region[mask_bright[:, :, 0]] = new_region[mask_bright[:, :, 0]]
            
            # 情况2: 混合
            blend_alpha = 0.6  # 新图权重
            blend_mask = overlap_mask & (~mask_bright)
            base_region[blend_mask[:, :, 0]] = (
                (1 - blend_alpha) * base_region[blend_mask[:, :, 0]].astype(np.float32) +
                blend_alpha * new_region[blend_mask[:, :, 0]].astype(np.float32)
            ).astype(np.uint8)

    def _save_frame_data(self, frame: np.ndarray, position: np.ndarray):
        """保存帧数据用于后续保存"""
        self.map_data['images'].append(frame.copy())
        self.map_data['positions'].append(position.copy())

    def reset(self):
        """重置拼接器"""
        self.stitched_map = None
        self.current_offset = np.array([0, 0], dtype=np.float32)
        self.map_data = {
            'images': [],
            'positions': []
        }
        self.last_frame = None
        self.transform_history.clear()
        self.frame_history.clear()

    def save_map(self, filepath: str):
        """保存拼接地图和数据"""
        try:
            if self.stitched_map is not None:
                # 保存图像
                cv2.imwrite(filepath, cv2.cvtColor(self.stitched_map, cv2.COLOR_RGB2BGR))

                # 保存数据
                data_filepath = filepath.replace('.png', '.dat').replace('.jpg', '.dat')
                with open(data_filepath, 'wb') as f:
                    pickle.dump(self.map_data, f)

                logger.info(f"地图已保存: {filepath}")
        except Exception as e:
            logger.error(f"保存地图失败: {e}")

    def load_map(self, filepath: str) -> bool:
        """加载已保存的地图"""
        try:
            # 加载图像
            img = cv2.imread(filepath)
            if img is not None:
                self.stitched_map = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 尝试加载数据
                data_filepath = filepath.replace('.png', '.dat').replace('.jpg', '.dat')
                try:
                    with open(data_filepath, 'rb') as f:
                        self.map_data = pickle.load(f)
                except:
                    pass

                logger.info(f"地图已加载: {filepath}")
                return True
        except Exception as e:
            logger.error(f"加载地图失败: {e}")
        return False
