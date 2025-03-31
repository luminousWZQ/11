"""ok
无人机控制模块测试套件
文件名：test_drone_controller.py
功能：验证无人机控制器的核心功能，支持模拟模式和真实硬件测试
"""

import json
import unittest
import time
from unittest.mock import Mock, patch
import cv2
import numpy as np
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_controller import DJICloudController, DroneState


class TestDroneController(unittest.TestCase):
    """主测试类"""

    # 测试配置（替换为实际参数进行真实测试）
    TEST_CONFIG = {
        "app_key": "test_key",
        "app_secret": "test_secret",
        "drone_sn": "TEST_SN_123",
        "mqtt_host": "test.mosquitto.org",
        "mqtt_port": 1883,
        "rtmp_url": "rtmp://test.stream.url"
    }

    def setUp(self):
        """初始化测试环境"""
        self.controller = DJICloudController(self.TEST_CONFIG)

        # 注入模拟MQTT客户端
        self.mock_client = Mock()
        self.controller.client = self.mock_client

        # 模拟视频流
        self.test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(self.test_frame, "SIMULATION FRAME", (400, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

    def test_initial_state(self):
        """测试初始状态"""
        self.assertFalse(self.controller.state.connected)
        self.assertEqual(self.controller.state.altitude, 0.0)
        self.assertEqual(self.controller.state.battery, 0)

    def test_successful_connection(self):
        """测试成功连接场景"""
        # 配置模拟连接成功
        self.mock_client.is_connected.return_value = True

        # 执行连接
        result = self.controller.connect(timeout=5)

        # 验证结果
        self.assertTrue(result)
        self.assertTrue(self.controller.state.connected)
        self.mock_client.connect.assert_called_once()
        self.mock_client.loop_start.assert_called_once()

    def test_failed_connection(self):
        """测试连接失败场景"""
        # 配置模拟连接失败
        self.mock_client.connect.side_effect = Exception("Connection timeout")

        # 执行连接
        result = self.controller.connect(timeout=1)

        # 验证结果
        self.assertFalse(result)
        self.assertFalse(self.controller.state.connected)

    def test_basic_commands(self):
        """测试基础指令发送"""
        # 配置模拟连接
        self.controller.state.connected = True

        # 测试起飞指令
        self.controller.takeoff()
        self.mock_client.publish.assert_called_once()

        # 验证指令参数
        _, kwargs = self.mock_client.publish.call_args
        payload = json.loads(kwargs['payload'])
        self.assertEqual(payload["method"], "takeoff")

        # 测试移动指令
        self.mock_client.reset_mock()
        self.controller.move_position(2, 3, 5)
        _, kwargs = self.mock_client.publish.call_args
        payload = json.loads(kwargs['payload'])
        self.assertEqual(payload["params"]["x"], 2)
        self.assertEqual(payload["params"]["y"], 3)
        self.assertEqual(payload["params"]["z"], 5)

    @patch('cv2.VideoCapture')
    def test_video_stream(self, mock_video):
        """测试视频流功能"""
        # 配置模拟视频流
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, self.test_frame)
        mock_video.return_value = mock_cap

        # 执行视频测试
        self.controller.start_video_stream()
        frame = self.controller.get_video_frame()
        self.controller.stop_video_stream()

        # 验证结果
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, self.test_frame.shape)
        mock_video.assert_called_once_with(self.TEST_CONFIG["rtmp_url"])
        mock_cap.release.assert_called_once()

    def test_emergency_procedures(self):
        """测试紧急流程"""
        # 模拟无人机已连接
        self.controller.state.connected = True  # <-- 新增此行

        # 配置健康告警
        test_payload = {
            "warnings": [{
                "level": 3,
                "message": "Critical battery level"
            }]
        }

        # 触发告警处理
        self.controller._handle_health_warnings(test_payload)

        # 验证紧急停止被调用
        self.mock_client.publish.assert_called_once()
        _, kwargs = self.mock_client.publish.call_args
        payload = json.loads(kwargs['payload'])
        self.assertEqual(payload["method"], "emergency_stop")

class RealDeviceTest(unittest.TestCase):
    """真实设备测试类（需要配置有效参数）"""

    REAL_CONFIG = {
        "app_key": "REAL_APP_KEY",
        "app_secret": "REAL_APP_SECRET",
        "drone_sn": "REAL_SERIAL_NUMBER",
        "mqtt_host": "cloud.dji.com",
        "mqtt_port": 8883,
        "rtmp_url": "rtmp://real.stream.url"
    }

    @unittest.skipUnless('REAL_APP_KEY' in locals(), "需要真实凭证")
    def test_real_connection(self):
        """真实设备连接测试"""
        controller = DJICloudController(self.REAL_CONFIG)
        try:
            connected = controller.connect(timeout=10)
            self.assertTrue(connected)
            self.assertIsNotNone(controller.get_current_gps())
        finally:
            controller.disconnect()


if __name__ == '__main__':
    # 执行测试
    unittest.main()