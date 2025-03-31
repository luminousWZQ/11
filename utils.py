"""
通用工具函数
功能：日志记录、数据保存、可视化等
"""
import logging
from datetime import datetime

def setup_logger():
    """配置日志系统"""
    logging.basicConfig(
        filename=f"log_{datetime.now().strftime('%Y%m%d')}.txt",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s"
    )

def save_results(data, filename):
    """保存检测结果到CSV"""
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def draw_annotations(frame, detections):
    """在图像上绘制检测框和注释"""
    annotated_frame = sv.BoxAnnotator().annotate(
        scene=frame.copy(),
        detections=detections
    )
    return annotated_frame