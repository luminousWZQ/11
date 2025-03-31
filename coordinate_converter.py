"""ok
坐标系转换模块
文件名：coordinate_converter.py
功能：实现像素坐标系到世界坐标系的完整转换链
依赖：numpy, opencv-python
"""

import numpy as np
import cv2
from typing import Tuple

class CoordinateConverter:
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray = None):
        """
        初始化坐标转换器
        :param camera_matrix: 相机内参矩阵 3x3
        :param dist_coeffs: 相机畸变系数 1x5 (可选)
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)

        # 预计算相机内参逆矩阵
        self.cam_matrix_inv = np.linalg.inv(self.camera_matrix)

    def undistort_points(self, pixel_points: np.ndarray) -> np.ndarray:
        """
        去除图像畸变 (可选项)
        :param pixel_points: 原始像素坐标 Nx2
        :return: 去畸变后的像素坐标 Nx2
        """
        if self.dist_coeffs is not None:
            undistorted = cv2.undistortPoints(
                pixel_points.astype(np.float32),
                self.camera_matrix,
                self.dist_coeffs,
                None,
                self.camera_matrix
            )
            return undistorted.reshape(-1, 2)  # 确保输出为 (N,2)
        return pixel_points

    def pixel_to_camera(self, pixel_coord: Tuple[float, float], depth: float) -> np.ndarray:
        """
        像素坐标转相机坐标系 (带畸变校正)
        :param pixel_coord: (u, v) 像素坐标
        :param depth: 目标深度值 (米)
        :return: 相机坐标系下的3D点 [x, y, z]^T
        """
        # 去畸变处理
        undistorted = self.undistort_points(np.array([pixel_coord]))
        u, v = undistorted[0]

        # 坐标归一化
        point_2d = np.array([u, v, 1])
        point_3d = self.cam_matrix_inv @ point_2d * depth
        return point_3d.reshape(3, 1)

    @staticmethod
    def camera_to_drone(camera_coord: np.ndarray) -> np.ndarray:
        """
        相机坐标系转无人机坐标系
        :param camera_coord: 相机坐标系下的点 [x, y, z]^T
        :return: 无人机坐标系下的点 [x, y, z]^T
        """
        # 大疆无人机坐标系转换矩阵 (需根据实际机型调整)
        R = np.array([[0, 1, 0],
                     [0, 0, 1],
                     [1, 0, 0]])
        return R @ camera_coord

    @staticmethod
    def drone_to_world(drone_coord: np.ndarray,
                      drone_pos: np.ndarray,
                      drone_attitude: np.ndarray) -> np.ndarray:
        """
        无人机坐标系转世界坐标系 (NED)
        :param drone_coord: 无人机坐标系下的点 [x, y, z]^T
        :param drone_pos: 无人机位置 [x, y, z] (米)
        :param drone_attitude: 无人机姿态 [roll, pitch, yaw] (弧度)
        :return: 世界坐标系下的点 [x, y, z]^T
        """
        # 解包姿态参数
        roll, pitch, yaw = drone_attitude

        # 构建旋转矩阵 (Z-Y-X顺序)
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])

        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])

        R_x = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])

        rotation_matrix = R_z @ R_y @ R_x

        # 坐标系转换
        return rotation_matrix @ drone_coord + drone_pos.reshape(3, 1)

    def pixel_to_world(self,
                      pixel_coord: Tuple[float, float],
                      depth: float,
                      drone_pos: np.ndarray,
                      drone_attitude: np.ndarray) -> np.ndarray:
        """
        完整坐标转换链
        :param pixel_coord: 像素坐标 (u, v)
        :param depth: 目标深度 (米)
        :param drone_pos: 无人机位置 [x, y, z] (米)
        :param drone_attitude: 无人机姿态 [roll, pitch, yaw] (弧度)
        :return: 世界坐标系坐标 [x, y, z]^T
        """
        # 转换到相机坐标系
        camera_coord = self.pixel_to_camera(pixel_coord, depth)

        # 转换到无人机坐标系
        drone_coord = self.camera_to_drone(camera_coord)

        # 转换到世界坐标系
        return self.drone_to_world(drone_coord, drone_pos, drone_attitude)

# 使用示例
if __name__ == "__main__":
    # 相机参数示例 (需实际标定)
    cam_matrix = np.array([
        [1200, 0, 960],
        [0, 1200, 540],
        [0, 0, 1]
    ])

    # 初始化转换器
    converter = CoordinateConverter(cam_matrix)

    # 示例参数
    pixel = (1000, 600)     # 像素坐标
    depth = 50.0            # 目标深度 (米)
    drone_pos = np.array([10, 20, 30])          # 无人机位置 (米)
    drone_attitude = np.radians([30, 15, 45])   # 姿态角转弧度

    # 执行转换
    world_coord = converter.pixel_to_world(pixel, depth, drone_pos, drone_attitude)
    print("世界坐标系坐标:", world_coord.flatten())