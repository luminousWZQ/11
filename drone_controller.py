"""
大疆无人机云API控制模块
文件名：drone_controller.py
功能：通过DJI Cloud API实现无人机实时控制、数据采集与视频流处理
依赖：paho-mqtt, opencv-python, numpy, requests
"""

import json
import time
import logging
import threading
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DroneController")


@dataclass
class DroneState:
    """无人机状态数据容器"""
    connected: bool = False
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0  # 相对高度（米）
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, z) 速度 m/s
    battery: int = 0  # 剩余电量百分比
    attitude: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (roll, pitch, yaw) 弧度


class DJICloudController:
    """大疆无人机云API控制器"""

    def __init__(self, config: Dict):
        """
        初始化控制器
        :param config: 配置字典，包含以下键：
            - app_key: 开发者应用密钥
            - app_secret: 开发者应用密钥
            - drone_sn: 无人机序列号
            - mqtt_host: MQTT服务器地址
            - mqtt_port: MQTT端口（默认8883）
            - rtmp_url: 视频流RTMP地址
        """
        self.config = config
        self.state = DroneState()
        self._command_queue = []
        self._video_cap = None
        self._setup_mqtt_client()

    def _setup_mqtt_client(self) -> None:
        """配置MQTT客户端连接"""
        self.client = mqtt.Client(transport="websockets")
        self.client.ws_set_options(
            path=f"/mqtt?device_sn={self.config['drone_sn']}"
        )
        self.client.username_pw_set(
            username=self.config["app_key"],
            password=self.config["app_secret"]
        )
        self.client.tls_set()  # 启用TLS加密

        # 注册回调函数
        self.client.on_connect = self._on_mqtt_connect
        self.client.on_message = self._on_mqtt_message

    def connect(self, timeout: int = 10) -> bool:
        """
        连接到无人机云服务
        :param timeout: 连接超时时间（秒）
        :return: 是否成功连接
        """
        try:
            self.client.connect(
                host=self.config["mqtt_host"],
                port=self.config.get("mqtt_port", 8883)
            )
            # 启动后台线程处理MQTT消息
            self.client.loop_start()

            # 等待连接建立
            start_time = time.time()
            while not self.state.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            return self.state.connected
        except Exception as e:
            logger.error(f"连接失败: {str(e)}")
            return False

    def _on_mqtt_connect(self, client, userdata, flags, rc) -> None:
        """MQTT连接回调"""
        if rc == 0:
            logger.info("成功连接到DJI云服务")
            self.state.connected = True
            # 订阅关键遥测主题
            self._subscribe_telemetry()
        else:
            logger.error(f"连接失败，错误码: {rc}")
            self.state.connected = False

    def _subscribe_telemetry(self) -> None:
        """订阅遥测数据主题"""
        topics = [
            f"thing/product/{self.config['drone_sn']}/osd",  # 基础状态
            f"thing/product/{self.config['drone_sn']}/hms",  # 健康管理
            f"thing/product/{self.config['drone_sn']}/camera"  # 相机状态
        ]
        for topic in topics:
            self.client.subscribe(topic, qos=1)
            logger.debug(f"已订阅主题: {topic}")

    def _on_mqtt_message(self, client, userdata, msg) -> None:
        """处理MQTT消息"""
        try:
            payload = json.loads(msg.payload)
            topic = msg.topic

            if "osd" in topic:
                # 更新基础状态
                self.state.latitude = payload.get("latitude", 0.0)
                self.state.longitude = payload.get("longitude", 0.0)
                self.state.altitude = payload.get("altitude", 0.0)
                self.state.velocity = (
                    payload.get("velocity_x", 0.0),
                    payload.get("velocity_y", 0.0),
                    payload.get("velocity_z", 0.0)
                )
                self.state.battery = payload.get("battery", 0)

            elif "hms" in topic:
                # 处理健康告警
                self._handle_health_warnings(payload)

            elif "camera" in topic:
                # 更新相机状态
                pass  # 可根据需要扩展

        except json.JSONDecodeError:
            logger.warning("收到无效JSON消息")

    def _handle_health_warnings(self, payload: Dict) -> None:
        """处理健康管理系统告警"""
        for warning in payload.get("warnings", []):
            level = warning.get("level", 0)
            message = warning.get("message", "")
            if level >= 2:  # 严重告警
                logger.error(f"健康告警: {message}")
                self.emergency_stop()
            else:
                logger.warning(f"健康警告: {message}")

    def send_command(self, command: str, params: Dict) -> bool:
        """
        发送控制指令
        :param command: 指令类型（如"takeoff", "land", "move"）
        :param params: 指令参数
        :return: 是否发送成功
        """
        if not self.state.connected:
            logger.warning("未连接无人机，指令被忽略")
            return False

        topic = f"thing/product/{self.config['drone_sn']}/commands"
        payload = {
            "method": command,
            "params": params,
            "timestamp": int(time.time() * 1000)
        }
        try:
            self.client.publish(topic, json.dumps(payload), qos=1)
            return True
        except Exception as e:
            logger.error(f"指令发送失败: {str(e)}")
            return False

    # ---------- 常用控制指令封装 ----------
    def takeoff(self) -> bool:
        """执行起飞"""
        return self.send_command("takeoff", {})

    def land(self) -> bool:
        """执行降落"""
        return self.send_command("land", {})

    def emergency_stop(self) -> bool:
        """紧急停止并降落"""
        return self.send_command("emergency_stop", {})

    def move_position(self, x: float, y: float, z: float) -> bool:
        """
        相对当前位置移动
        :param x: 东方向位移（米）
        :param y: 北方向位移（米）
        :param z: 垂直方向位移（米）
        """
        return self.send_command("move", {
            "x": x,
            "y": y,
            "z": z,
            "coord_type": "body"  # 机体坐标系
        })

    def set_speed(self, speed: float) -> bool:
        """设置最大飞行速度（米/秒）"""
        return self.send_command("set_max_speed", {"speed": speed})

    # ---------- 视频流处理 ----------
    def start_video_stream(self) -> None:
        """启动视频流捕获"""
        if self._video_cap is None:
            self._video_cap = cv2.VideoCapture(self.config["rtmp_url"])
            if not self._video_cap.isOpened():
                logger.error("无法打开视频流")
                self._video_cap = None

    def get_video_frame(self) -> Optional[np.ndarray]:
        """获取当前视频帧"""
        if self._video_cap and self._video_cap.isOpened():
            ret, frame = self._video_cap.read()
            return frame if ret else None
        return None

    def stop_video_stream(self) -> None:
        """停止视频流"""
        if self._video_cap:
            self._video_cap.release()
            self._video_cap = None

    # ---------- 其他功能 ----------
    def get_current_gps(self) -> Tuple[float, float]:
        """获取当前GPS坐标（纬度, 经度）"""
        return (self.state.latitude, self.state.longitude)

    def disconnect(self) -> None:
        """断开连接并清理资源"""
        self.client.loop_stop()
        self.client.disconnect()
        self.stop_video_stream()
        self.state.connected = False
        logger.info("已断开无人机连接")


# 使用示例
if __name__ == "__main__":
    config = {
        "app_key": "your_app_key",
        "app_secret": "your_app_secret",
        "drone_sn": "your_drone_sn",
        "mqtt_host": "cloud.dji.com",
        "mqtt_port": 8883,
        "rtmp_url": "rtmp://cloud.dji.com/live/your_stream_key"
    }

    controller = DJICloudController(config)

    if controller.connect():
        try:
            # 基础控制测试
            if controller.takeoff():
                time.sleep(5)
                controller.move_position(2, 0, 5)  # 向东2米，上升5米
                time.sleep(8)
                controller.land()

            # 视频流测试
            controller.start_video_stream()
            start_time = time.time()
            while time.time() - start_time < 10:
                frame = controller.get_video_frame()
                if frame is not None:
                    cv2.imshow("Drone View", frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
            cv2.destroyAllWindows()

        finally:
            controller.disconnect()
    else:
        logger.error("无法连接到无人机")