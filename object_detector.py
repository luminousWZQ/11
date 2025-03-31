"""
目标检测与跟踪模块
文件名：object_detector.py
功能：基于YOLOv8的实时目标检测与ByteTrack多目标跟踪
依赖库：ultralytics==8.0.0, supervision==0.1.0, opencv-python>=4.5.0  supervision==0.24.0
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from ultralytics import YOLO
import supervision as sv


class ObjectDetector:
    """
    目标检测与跟踪核心类
    实现功能：
    - YOLOv8模型加载与推理
    - ByteTrack多目标跟踪
    - 检测结果过滤与后处理
    - 实时标注渲染
    """

    def __init__(self,
                 model_path: str,
                 target_class_id: int = 0,
                 confidence_threshold: float = 0.5,
                 tracking_config: Optional[Dict[str, Any]] = None):

        """
        初始化检测跟踪器

        参数：
            model_path (str): YOLOv8模型文件路径(.pt格式)
            target_class_id (int): 需要跟踪的目标类别ID，默认为0
            confidence_threshold (float): 检测置信度阈值(0-1)，默认0.5
            tracking_config (dict): ByteTrack跟踪器配置字典，可选参数：
                track_thresh (float): 跟踪阈值(默认0.25)
                track_buffer (int): 跟踪缓冲区大小(默认30)
                match_thresh (float): 匹配阈值(默认0.8)
                frame_rate (int): 帧率(默认30)
        """
        # 模型加载与验证
        self.model = self._load_model(model_path)
        self.target_class_id = target_class_id
        self.conf_thresh = confidence_threshold

        # 初始化ByteTrack跟踪器
        # self.tracker = sv.ByteTrack(
        #     **(tracking_config or self._default_tracking_config())
        # )
        self.tracker = sv.ByteTrack()  # 直接初始化，不使用自定义参数
        #

        # 初始化标注工具
        self.box_annotator = sv.BoxAnnotator(
            color=sv.ColorPalette.DEFAULT,  # 框颜色来源（动态分配）
            thickness=2,
            text_scale=0.6,  # 修正参数名
            text_padding=5,
            text_color=sv.Color.BLACK,
            color_lookup=sv.ColorPalette.DEFAULT
        )

    def _default_tracking_config(self) -> Dict[str, Any]:
        return {
            'track_activation_threshold': 0.25,  # 替代 track_thresh
            'lost_track_buffer': 30,  # 替代 track_buffer
            'matching_threshold': 0.8,  # 替代 match_thresh
            'frame_rate': 30
        }

    def _load_model(self, model_path: str) -> YOLO:
        """
        加载并验证YOLOv8模型

        参数：
            model_path (str): 模型文件路径

        返回：
            YOLO: 加载成功的模型对象

        异常：
            FileNotFoundError: 模型文件不存在
            RuntimeError: 模型加载失败
        """
        # 检查文件是否存在
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        try:
            model = YOLO(model_path, task="detect")
            if not model.names:
                raise ValueError("无效的模型文件结构")
            print(f"成功加载模型，检测类别: {model.names}")
            return model
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def detect(self, frame: np.ndarray) -> Tuple[sv.Detections, np.ndarray]:
        """
        执行目标检测与跟踪的全流程处理

        参数：
            frame (np.ndarray): 输入BGR图像帧，形状(H, W, 3)

        返回：
            Tuple[sv.Detections, np.ndarray]:
                - 检测结果对象(sv.Detections)
                - 标注后的BGR图像帧
        """
        # YOLOv8推理
        results = self.model(frame, verbose=False)[0]

        # 转换为supervision格式
        detections = sv.Detections.from_ultralytics(results)

        # 应用置信度过滤
        detections = detections[detections.confidence > self.conf_thresh]

        # 更新跟踪器状态
        detections = self.tracker.update_with_detections(detections)

        # 过滤指定类别
        target_detections = detections[detections.class_id == self.target_class_id]

        # 生成标注信息
        labels = self._generate_labels(target_detections)
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=target_detections,
            labels=labels
        )

        return target_detections, annotated_frame

    def _generate_labels(self, detections: sv.Detections) -> list:
        """
        生成检测标签文本

        参数：
            detections (sv.Detections): 检测结果对象

        返回：
            list: 包含标签字符串的列表
        """
        labels = []
        for tracker_id, class_id, confidence in zip(
            detections.tracker_id,
            detections.class_id,
            detections.confidence
        ):
            class_name = self.model.names[class_id]
            labels.append(
                f"#{tracker_id} {class_name} {confidence:.2f}"
            )
        return labels

    def filter_detections(
        self,
        detections: sv.Detections,
        min_area: int = 1000,
        max_aspect_ratio: float = 3.0,
        edge_margin: int = 20
    ) -> sv.Detections:
        """
        过滤检测结果（可选后处理）

        参数：
            detections (sv.Detections): 原始检测结果
            min_area (int): 最小边界框面积(像素)
            max_aspect_ratio (float): 最大宽高比(宽度/高度)
            edge_margin (int): 距图像边缘的最小像素距离

        返回：
            sv.Detections: 过滤后的检测结果
        """
        # 计算几何参数
        img_height, img_width = detections.xyxy[:, 3], detections.xyxy[:, 2]
        widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
        heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
        areas = widths * heights
        aspect_ratios = widths / (heights + 1e-6)  # 防止除以零

        # 创建过滤掩码
        valid_area = areas > min_area
        valid_aspect = aspect_ratios < max_aspect_ratio
        valid_position = (
            (detections.xyxy[:, 0] > edge_margin) &
            (detections.xyxy[:, 1] > edge_margin) &
            (detections.xyxy[:, 2] < img_width - edge_margin) &
            (detections.xyxy[:, 3] < img_height - edge_margin)
        )

        return detections[valid_area & valid_aspect & valid_position]

# 使用示例
if __name__ == "__main__":
    # 初始化检测器
    detector = ObjectDetector(
        model_path="D:/gdwangjk/project_root/models/best.pt",
        tracking_config={
            'track_activation_threshold': 0.3,  # 使用新参数名
            'lost_track_buffer': 45
        }
    )

    # 视频处理示例
    cap = cv2.VideoCapture("D:/gdwangjk/project_root/tests/test_video.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 执行检测与跟踪
        detections, annotated_frame = detector.detect(frame)

        # 可选：应用附加过滤
        filtered_detections = detector.filter_detections(detections)

        # 显示结果
        cv2.imshow("Detection Results", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()