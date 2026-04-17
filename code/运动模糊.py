# -*- coding: utf-8 -*-
import os
import cv2
import random
import numpy as np
from glob import glob

# =========================
# 1. 路径配置
# =========================
SRC_DIR = r"/home/xgq/Desktop/天气光照实验/原始/images"      # 原始图片文件夹
DST_DIR = r"/home/xgq/Desktop/天气光照实验/运动模糊/55模糊"   # 输出文件夹

# 支持的图片后缀
IMG_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]

# =========================
# 2. 模糊配置
# =========================
# 这里只保留一种更重的运动模糊
KERNEL_SIZE = 55   # 越大越模糊，建议 35/45/55 试试

# 是否随机角度
RANDOM_ANGLE = True

# 如果不随机角度，可以固定一个角度（单位：度）
FIXED_ANGLE = 45

# 是否保留原图到输出目录
COPY_ORIGINAL = True

# 随机种子（想复现可固定）
random.seed(42)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_all_images(src_dir):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob(os.path.join(src_dir, ext)))
    files.sort()
    return files


def motion_blur_kernel(kernel_size=45, angle=0):
    """
    生成运动模糊卷积核
    :param kernel_size: 核大小，建议奇数
    :param angle: 模糊角度，单位度
    :return: 归一化后的卷积核
    """
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    # 中间一条水平线
    kernel[(kernel_size - 1) // 2, :] = np.ones(kernel_size, dtype=np.float32)

    # 旋转卷积核
    center = (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (kernel_size, kernel_size))

    # 归一化
    s = kernel.sum()
    if s > 0:
        kernel /= s

    return kernel


def apply_motion_blur(img, kernel_size=45, angle=0):
    """
    对图像施加更强的运动模糊
    """
    kernel = motion_blur_kernel(kernel_size, angle)
    blurred = cv2.filter2D(img, -1, kernel)

    blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    return blurred


def choose_angle():
    """
    随机生成角度
    """
    return random.uniform(0, 180)


def save_image(path, img):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:
        cv2.imwrite(path, img)


def main():
    ensure_dir(DST_DIR)
    ensure_dir(os.path.join(DST_DIR, "heavy_blur"))

    if COPY_ORIGINAL:
        ensure_dir(os.path.join(DST_DIR, "original"))

    img_paths = get_all_images(SRC_DIR)
    print(f"共找到 {len(img_paths)} 张图片")

    if len(img_paths) == 0:
        print("未找到图片，请检查 SRC_DIR 路径是否正确。")
        return

    for idx, img_path in enumerate(img_paths, 1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[跳过] 无法读取: {img_path}")
            continue

        base_name = os.path.basename(img_path)

        # 保存原图
        if COPY_ORIGINAL:
            save_image(os.path.join(DST_DIR, "original", base_name), img)

        # 选择角度
        angle = choose_angle() if RANDOM_ANGLE else FIXED_ANGLE

        # 应用重度运动模糊
        blur_img = apply_motion_blur(img, kernel_size=KERNEL_SIZE, angle=angle)

        save_path = os.path.join(DST_DIR, "heavy_blur", base_name)
        save_image(save_path, blur_img)

        print(f"[{idx}/{len(img_paths)}] 已处理: {base_name} | angle={angle:.2f}")

    print("全部处理完成！")
    print(f"输出目录: {DST_DIR}")


if __name__ == "__main__":
    main()