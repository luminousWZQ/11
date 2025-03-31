"""ok
相机标定脚本
功能：通过棋盘格图像计算相机内参矩阵和畸变系数
"""

import cv2
import numpy as np
import glob
import os
import time
from datetime import  datetime

"""
cv2：用于OpenCV图像处理和操作。
numpy：用于数值计算和数据处理。
glob和os：用于文件路径管理和操作系统任务。
"""

cv2.ocl.setUseOpenCL(True)  # 启用OpenCL加速

# ==================== 配置参数 ====================
chessboard_ColCornerCount = 10  # 棋盘格列角点数
chessboard_RowCornerCount = 7   # 棋盘格行角点数
square_Size = 0.025            # 单个棋盘格实际尺寸(米)
calib_ImagesPath = r"D:\gdwangjk\project_root\chessboard\02\*.jpg"  # 标定图片路径,读取所有以.jpg为后缀的文件
output_FileName = r"D:\gdwangjk\project_root\config\camera_params.npz"  # 输出文件名
show_Corners = False             # 是否显示角点检测过程 (原v1_2.py中的可视化逻辑)
save_debug_images = False         # 新增：是否保存中间图像
debug_output_dir = r"D:\gdwangjk\project_root\output\calibration_debug"  # 中间图像保存路径
# ===================================================================

def create_debug_dir():
    """创建调试图像输出目录"""
    if not os.path.exists(debug_output_dir):
        os.makedirs(debug_output_dir)
        print(f"已创建调试目录：{debug_output_dir}")


def save_debug_image(image, step_name, idx):
    """保存调试图像
    :param image: 要保存的图像数据
    :param step_name: 处理阶段名称（original/gray/corners等）
    :param idx: 图像序号
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{idx:03d}_{step_name}_{timestamp}.jpg"
    output_path = os.path.join(debug_output_dir, filename)
    cv2.imwrite(output_path, image)
    print(f"已保存调试图像：{filename}")


def enhance_image_contrast(img):
    """图像对比度增强预处理"""
    # 转换为HSV空间处理亮度通道
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # CLAHE自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v = clahe.apply(v)

    # 合并通道并转回BGR
    enhanced_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)


def auto_detect_chessboard(gray_img, pattern_size, image_Path):
    """多尺度自动检测棋盘格"""
    start_time = time.time()
    scale_factors = [1.0, 0.75, 0.5]

    for scale in scale_factors:
        # 调整图像尺寸
        scaled_img = cv2.resize(gray_img, None, fx=scale, fy=scale)

        # 优化检测参数组合
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                 cv2.CALIB_CB_NORMALIZE_IMAGE )

        # 执行检测
        ret, corners = cv2.findChessboardCorners(scaled_img, pattern_size, flags=flags)
        print(f"尺度 {scale}x 检测耗时: {time.time() - start_time:.2f}s")
        if ret:
            # 将坐标还原到原始尺寸
            corners = corners.astype(np.float32) / np.array([[scale, scale]])
            # 检查角点有效性
            if corners.size == 0:
                print(f"警告: 图片 {os.path.basename(image_Path)} 检测到空角点数组")
                continue
            return corners
    return None

def main():
    """
    主函数：执行完整的相机标定流程
    流程说明：
    1. 准备物体坐标系点
    2. 遍历所有标定图片检测角点
    3. 执行相机标定计算参数
    4. 保存结果并验证标定效果
    """

    create_debug_dir()  # 确保目录存在

    # 准备物体点：(0,0,0), (1,0,0), ..., (chessboard_ColCornerCount-1, chessboard_RowCornerCount-1,0)
    objp = np.zeros((chessboard_ColCornerCount*chessboard_RowCornerCount, 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_ColCornerCount, 0:chessboard_RowCornerCount].T.reshape(-1,2)
    objp *= square_Size  # 缩放至实际物理尺寸

    # 存储物体点和图像点的列表
    vectorObjectPoint = []  # 三维物体点列表
    vectorImagePoint = []   # 二维图像点列表

    # 获取标定图片列表 (与v1_2.py中的图片遍历逻辑对应)
    calib_Images = glob.glob(calib_ImagesPath)
    if not calib_Images:
        print(f"[错误] 在 {calib_ImagesPath} 路径下未找到标定图片!")
        return

    valid_ImageCount = 0  # 有效标定图片计数器 (对应v1_2.py中的patternWasFound判断)

    # 处理每张标定图片
    for idx, image_Path in enumerate(calib_Images, 1):
        print(f"正在处理第 {idx}/{len(calib_Images)} 张图片: {os.path.basename(image_Path)}")
        # 读取图片
        src_Mat = cv2.imread(image_Path)
        if src_Mat is None:
            print(f"[警告] 无法读取图片: {os.path.basename(image_Path)}")
            continue
        # cv2.imshow(f"chessboard{idx}", src_Mat)
        # cv2.waitKey(0)

        if save_debug_images:
            save_debug_image(src_Mat, "01_original", idx)

        # 转换为灰度图 (对应v1_2.py中的grayMat)
        # gray_Mat = cv2.cvtColor(src_Mat, cv2.COLOR_BGR2GRAY)
        enhanced_Mat = enhance_image_contrast(src_Mat)
        gray_Mat = cv2.cvtColor(enhanced_Mat, cv2.COLOR_BGR2GRAY)

        if save_debug_images:
            save_debug_image(gray_Mat, "02_gray", idx)
        # cv2.imshow(f"chessboard_gray{idx}", gray_Mat)
        # cv2.waitKey(0)

        #查找棋盘格角点 (与v1_2.py中的cv2.findChessboardCorners调用一致)
        """
        参数flag，用于指定在检测棋盘格角点的过程中所应用的一种或多种过滤方法，可以使用下面的一种或多种，如果都是用则使用OR：
        cv::CALIB_CB_ADAPTIVE_THRESH：使用自适应阈值将图像转化成二值图像
        cv::CALIB_CB_NORMALIZE_IMAGE：归一化图像灰度系数(用直方图均衡化或者自适应阈值)
        /cv::CALIB_CB_FILTER_QUADS：在轮廓提取阶段，使用附加条件排除错误的假设
        /cv::CALIB_CV_FAST_CHECK：快速检测
        """
        # patternFound, corners = cv2.findChessboardCorners(
        #     gray_Mat,
        #     (chessboard_ColCornerCount, chessboard_RowCornerCount),
        #     flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        # )

        # 多尺度自动检测
        pattern_size = (chessboard_ColCornerCount, chessboard_RowCornerCount)
        corners = auto_detect_chessboard(gray_Mat, pattern_size, image_Path)
        #
        # # 检查角点有效性
        # if corners.size == 0:
        #     print(f"警告: 图片 {os.path.basename(image_Path)} 检测到空角点数组")
        #     continue

        # 强制规范数据类型和形状03
        corners = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)

        # 检查坐标范围
        h, w = gray_Mat.shape
        valid_corners = []
        for pt in corners:
            x, y = pt.ravel()
            if 0 <= x < w and 0 <= y < h:
                valid_corners.append(pt)
            else:
                print(f"无效坐标: ({x:.1f}, {y:.1f})")

        if not valid_corners:
            print(f"错误: 图片 {os.path.basename(image_Path)} 无有效角点")
            continue

        corners = np.array(valid_corners, dtype=np.float32).reshape(-1, 1, 2)
        print(f"角点数据类型: {corners.dtype} | 形状: {corners.shape}")
        print(f"图像类型: {gray_Mat.dtype} | 尺寸: {gray_Mat.shape}")

        if corners is not None:
            # 亚像素级角点优化 (与v1_2.py中的cv2.cornerSubPix逻辑一致) criteria (type, max_iter, epsilon)(终止条件的类型，最大迭代次数，精度阈值)
            criteria = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)    # 定义亚像素角点迭代终止条件，同时满足精度 (EPS) 和最大迭代次数 (MAX_ITER) 条件
            corners_Refined = cv2.cornerSubPix(gray_Mat, corners, (11,11), (-1,-1), criteria) #   # 进一步提取亚像素角点

            # 保存角点检测结果
            if save_debug_images:
                raw_corner_img = src_Mat.copy()
                cv2.drawChessboardCorners(raw_corner_img, pattern_size, corners, True)
                save_debug_image(raw_corner_img, "03_raw_corners", idx)

                refined_img = src_Mat.copy()
                cv2.drawChessboardCorners(refined_img, pattern_size, corners_Refined, True)
                save_debug_image(refined_img, "04_refined_corners", idx)

            # 输出处理后的角点坐标
            # print("Processed corner points:")
            # for corner in corners:
            # print(corner)

            # 存储物体点和图像点 (对应v1_2.py中的vectorObjectPoint/vectorImagePoint)
            vectorObjectPoint.append(objp)
            vectorImagePoint.append(corners_Refined)
            valid_ImageCount += 1

            # 实时显示
            if show_Corners:
                display_img = cv2.resize(refined_img, (1280, 720))
                cv2.imshow('角点检测结果', display_img)
                cv2.waitKey(500)

            # # 可视化角点检测结果 (对应v1_2.py中的绘制逻辑)
            # if show_Corners:
            #     draw_Mat = src_Mat.copy()
            #     cv2.drawChessboardCorners(
            #         draw_Mat,
            #         (chessboard_ColCornerCount, chessboard_RowCornerCount),
            #         corners_Refined,
            #         patternFound
            #     )
            #     cv2.putText(draw_Mat, f"number: {valid_ImageCount}", (10,30),
            #                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            #     cv2.imshow('角点检测', draw_Mat)
            #     cv2.waitKey(500)  # 显示0.5秒
            # else:
            #     print(f"自动检测失败: {os.path.basename(image_Path)}")

    cv2.destroyAllWindows()     # 释放资源

    # 标定参数有效性检查 (比v1_2.py增加错误处理)
    if valid_ImageCount < 10:
        print(f"[错误] 有效标定图片不足 ({valid_ImageCount}张)，至少需要10张")
        return

    # 执行相机标定 (与v1_2.py中的cv2.calibrateCamera调用一致)
    print(f"\n开始标定，使用 {valid_ImageCount} 张有效图片...")
    ret, cam_Matrix, cam_DistCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        vectorObjectPoint,
        vectorImagePoint,
        gray_Mat.shape[::-1],  # 图像尺寸 (width, height)
        None,
        None
    )

    # 标定结果验证 (v1_2.py原有打印逻辑增强)
    print("\n标定结果：")
    print(f"重投影误差：{ret:.4f} (值应小于0.5)")
    print("内参矩阵(cam_Matrix)：\n", cam_Matrix)
    print("畸变系数(cam_DistCoeffs)：", cam_DistCoeffs.flatten())

    # 保存标定结果 (对应v1_2.py中的参数保存需求)
    np.savez(
        output_FileName,
        cam_Matrix=cam_Matrix,
        cam_DistCoeffs=cam_DistCoeffs,
        rvecs=rvecs,
        tvecs=tvecs
    )
    print(f"\n标定参数已保存至：{output_FileName}")

    # 去畸变效果验证 (比v1_2.py增加可视化验证)
    if valid_ImageCount > 0:
        test_Image = cv2.imread(calib_Images[0])
        h, w = test_Image.shape[:2]

        # 计算最优新相机矩阵
        new_CamMatrix, roi = cv2.getOptimalNewCameraMatrix(
            cam_Matrix,
            cam_DistCoeffs,
            (w, h),
            1,
            (w, h)
        )

        # 去畸变处理 (对应v1_2.py中的initUndistortRectifyMap逻辑)
        map1, map2 = cv2.initUndistortRectifyMap(
            cam_Matrix,
            cam_DistCoeffs,
            None,
            new_CamMatrix,
            (w, h),
            cv2.CV_32FC1
        )
        undistorted_Image = cv2.remap(test_Image, map1, map2, cv2.INTER_LINEAR)

        # 并排显示对比结果
        comparison = np.hstack((test_Image, undistorted_Image))
        if save_debug_images:
            save_debug_image(comparison, "05_undistorted_compare", 0)

            # 显示结果
            cv2.imshow("原始 vs 去畸变", cv2.resize(comparison, (1280, 720)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()