"""
无人机视觉测距与抵近时间预测系统 - 距离计算模块
版本: 2.1
核心功能：
1. 多目标运动状态跟踪
2. 实时距离解算
3. 自适应运动预测
4. 系统状态持久化
5. 异常恢复机制
"""

# ==================== 导入依赖 ====================
import numpy as np
import pandas as pd
import os
import logging
import shutil
from threading import Lock
from time import perf_counter
from typing import Dict, Optional, Tuple
from numba import njit  # 用于加速数值计算

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system.log"),  # 文件日志
        logging.StreamHandler()             # 控制台日志
    ]
)
logger = logging.getLogger("DistanceModule")

# ==================== 环形缓冲区类 ====================
class RingBuffer:
    """
    线程安全的高性能环形缓冲区
    特性：
    - 固定内存预分配
    - 支持正向/反向数据访问
    - 自动覆盖旧数据

    参数:
        size (int): 缓冲区容量
    """

    def __init__(self, size: int):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float64)  # 存储容器
        self.index = 0      # 当前写入位置
        self.count = 0      # 有效数据计数
        self.lock = Lock()  # 线程安全锁

    def append(self, value: float) -> None:
        """追加新数据（线程安全）"""
        with self.lock:
            self.buffer[self.index] = value
            self.index = (self.index + 1) % self.size  # 环形指针
            self.count = min(self.count + 1, self.size)

    def get_data(self, reverse: bool = False) -> np.ndarray:
        """
        获取有效数据数组
        参数:
            reverse (bool): 是否反向获取（最新数据在前）
        返回:
            np.ndarray: 数据数组
        """
        with self.lock:
            if self.count < self.size:
                data = self.buffer[:self.count]
            else:
                # 通过滚动索引拼接数据
                data = np.roll(self.buffer, -self.index)[:self.size]
            return data[::-1] if reverse else data

# ==================== JIT加速函数 ====================
@njit
def fast_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    JIT加速的三维欧氏距离计算
    参数:
        a, b (np.ndarray): 形状为(3,)的坐标数组
    返回:
        float: 两点间距离（米）
    """
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

@njit
def adaptive_smoothing(data: np.ndarray, base_alpha: float) -> float:
    """
    自适应指数平滑算法
    特性:
    - 数据点越少平滑力度越大
    - 自动抑制异常波动

    参数:
        data (np.ndarray): 输入数据（时间倒序）
        base_alpha (float): 基础平滑系数(0-1)
    返回:
        float: 平滑后的预测值
    """
    if len(data) == 0:
        return np.nan

    # 动态调整系数
    effective_alpha = min(base_alpha, 2/(len(data)+1))
    smoothed = data[0]
    for value in data[1:]:
        smoothed = effective_alpha * value + (1 - effective_alpha) * smoothed
    return smoothed

# ==================== 主计算类 ====================
class EnhancedDistanceCalculator:
    """
    增强型距离计算器
    功能架构：
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │ 数据采集层   │ ==> │ 数据处理层   │ ==> │ 预测输出层   │
    └─────────────┘     └─────────────┘     └─────────────┘
    """

    def __init__(self,
                 fixed_point: np.ndarray,
                 buffer_size: int = 60,
                 speed_limits: Tuple[float, float] = (1.0, 83.3),
                 auto_save_interval: int = 300):
        """
        初始化计算器
        参数:
            fixed_point (np.ndarray): 施工点世界坐标[x,y,z]
            buffer_size (int): 每个目标的环形缓冲区容量
            speed_limits (tuple): 有效速度范围(m/s)
            auto_save_interval (int): 自动保存间隔(秒)
        """
        # 坐标系参数
        self.fixed_point = fixed_point.astype(np.float64)  # 强制双精度

        # 计算参数
        self.buffer_size = buffer_size
        self.speed_limits = speed_limits
        self.auto_save_interval = auto_save_interval

        # 数据存储
        self.speed_buffers: Dict[int, RingBuffer] = {}       # 目标速度缓冲区
        self.last_valid_positions: Dict[int, np.ndarray] = {}  # 最后有效位置
        self.last_update_times: Dict[int, float] = {}        # 最后更新时间

        # 状态管理
        self.data_lock = Lock()                             # 全局数据锁
        self.system_state_file = "system_state.pkl"         # 状态文件
        self.last_save_time = perf_counter()                # 最后保存时间

        self._load_state()  # 加载历史状态

    def update_position(self, tracker_id: int, current_pos: np.ndarray) -> None:
        """
        更新目标位置（核心方法）
        处理流程:
        1. 数据有效性验证
        2. 速度计算
        3. 数据存储
        4. 状态保存
        """
        try:
            # ==== 数据校验 ====
            if not np.isfinite(current_pos).all():
                logger.warning(f"目标{tracker_id}无效坐标: {current_pos}")
                return

            # ==== 数据准备 ====
            current_time = perf_counter()

            with self.data_lock:
                # 初始化新目标
                if tracker_id not in self.speed_buffers:
                    self.speed_buffers[tracker_id] = RingBuffer(self.buffer_size)
                    self.last_update_times[tracker_id] = current_time

                # 计算时间差
                last_time = self.last_update_times.get(tracker_id, current_time)
                delta_time = current_time - last_time

                # ==== 速度计算 ====
                last_pos = self.last_valid_positions.get(tracker_id)
                if last_pos is not None and delta_time > 1e-6:
                    displacement = fast_distance(current_pos, last_pos)
                    speed = displacement / delta_time

                    # 速度滤波
                    if self.speed_limits[0] <= speed <= self.speed_limits[1]:
                        self.speed_buffers[tracker_id].append(speed)
                    else:
                        logger.debug(f"目标{tracker_id}异常速度: {speed:.2f}m/s")

                # 更新状态
                self.last_valid_positions[tracker_id] = current_pos.copy()
                self.last_update_times[tracker_id] = current_time

            # ==== 自动保存 ====
            if current_time - self.last_save_time > self.auto_save_interval:
                self._save_state()
                self.last_save_time = current_time

        except Exception as e:
            logger.error(f"位置更新异常: {str(e)}")
            self._recover_state()

    def calculate_eta(self, tracker_id: int) -> Optional[float]:
        """
        计算抵近时间(ETA)
        算法流程:
        1. 获取当前距离
        2. 计算平滑速度
        3. 时间预测
        """
        try:
            with self.data_lock:
                current_pos = self.last_valid_positions.get(tracker_id)
                if current_pos is None:
                    return None

                # 实时距离
                distance = fast_distance(current_pos, self.fixed_point)

                # 速度处理
                buffer = self.speed_buffers.get(tracker_id)
                if buffer and buffer.count > 5:  # 最小数据量要求
                    data = buffer.get_data(reverse=True)[:30]  # 取最近30个点
                    speed = adaptive_smoothing(data, 0.2)      # 基础平滑系数0.2

                    if speed > 0.1:  # 忽略微小速度
                        return distance / speed
                return None
        except Exception as e:
            logger.error(f"ETA计算失败: {str(e)}")
            return None

    # ==================== 状态管理 ====================
    def _save_state(self) -> None:
        """保存系统状态（含备份机制）"""
        with self.data_lock:
            state = {
                'fixed_point': self.fixed_point,
                'last_positions': self.last_valid_positions,
                'buffer_data': {tid: buf.get_data() for tid, buf in self.speed_buffers.items()}
            }
            try:
                # 创建备份
                if os.path.exists(self.system_state_file):
                    shutil.copy(self.system_state_file, self.system_state_file+".bak")
                # 保存状态
                pd.to_pickle(state, self.system_state_file)
                logger.info("系统状态已保存")
            except Exception as e:
                logger.error(f"状态保存失败: {str(e)}")

    def _load_state(self) -> None:
        """加载系统状态"""
        if os.path.exists(self.system_state_file):
            try:
                state = pd.read_pickle(self.system_state_file)
                # 校验关键字段
                required_keys = {'fixed_point', 'last_positions', 'buffer_data'}
                if not required_keys.issubset(state.keys()):
                    raise ValueError("状态文件损坏")

                with self.data_lock:
                    self.fixed_point = state['fixed_point']
                    self.last_valid_positions = state['last_positions']
                    # 重建缓冲区
                    self.speed_buffers = {}
                    for tid, data in state['buffer_data'].items():
                        self.speed_buffers[tid] = RingBuffer(self.buffer_size)
                        for val in data:
                            self.speed_buffers[tid].append(val)
                logger.info("历史状态已加载")
            except Exception as e:
                logger.error(f"状态加载失败: {str(e)}")
                self._load_backup()

    def _load_backup(self) -> None:
        """从备份文件恢复"""
        backup_file = self.system_state_file + ".bak"
        if os.path.exists(backup_file):
            try:
                state = pd.read_pickle(backup_file)
                pd.to_pickle(state, self.system_state_file)
                self._load_state()
            except Exception as e:
                logger.error("备份恢复失败，初始化空状态")
                self._init_empty_state()

    def _init_empty_state(self) -> None:
        """初始化空状态"""
        with self.data_lock:
            self.last_valid_positions = {}
            self.speed_buffers = {}

    def _recover_state(self) -> None:
        """异常恢复流程"""
        logger.warning("执行异常恢复...")
        self._load_state()

# ==================== 测试用例 ====================
if __name__ == "__main__":
    # 初始化测试环境
    test_site = np.array([120.5, 245.3, 0.0], dtype=np.float64)
    calculator = EnhancedDistanceCalculator(
        fixed_point=test_site,
        buffer_size=60,
        speed_limits=(2.0, 50.0),
        auto_save_interval=5
    )

    # 模拟数据流（X轴匀速运动）
    for frame in range(200):
        try:
            target_id = 1001
            pos = np.array([frame*1.0, 200.0, 0.0], dtype=np.float64)

            # 更新位置
            calculator.update_position(target_id, pos)

            # 定期计算ETA
            if frame % 10 == 0:
                eta = calculator.calculate_eta(target_id)
                if eta:
                    print(f"帧 {frame:03d}: ETA {eta:.2f}秒")

            # 模拟异常
            if frame == 150:
                raise RuntimeError("模拟传感器故障")

        except Exception as e:
            logger.error(f"测试异常: {str(e)}")
            calculator._recover_state()

    # 最终保存
    calculator._save_state()
    logger.info("测试完成")