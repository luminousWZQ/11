import cv2
import yaml
from drone.cloud_controller import DJICloudController
from vision.object_detector import TrainDetector
from vision.coordinate import GeoConverter

class Application:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # 初始化组件
        self.drone = DJICloudController(self.config['drone'])
        self.detector = TrainDetector(self.config['model_path'])
        self.converter = GeoConverter(
            np.load(self.config['camera_params'])['cam_Matrix'],
            np.load(self.config['camera_params'])['cam_DistCoeffs']
        )

    def run(self):
        while True:
            # 获取实时数据
            frame = self.drone.get_video_stream()
            if frame is None:
                continue

            # 目标检测
            detections = self.detector.detect(frame)

            # 处理每个检测目标
            for detection in detections:
                if detection.class_id != 0:  # 0为列车类别
                    continue

                # 计算中心点坐标
                bbox = detection.xyxy[0]
                center = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)

                # 坐标转换
                geo_position = self.converter.pixel_to_geo(
                    center, self.drone.state)

                # 计算与目标点的距离
                distance = geodesic(
                    geo_position,
                    self.config['target_position']
                ).meters

                # 显示结果
                self._display_info(frame, geo_position, distance)

            # 实时显示
            cv2.imshow("Drone View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    app = Application("config/config.yaml")
    app.run()