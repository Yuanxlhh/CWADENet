import os
import glob
import shutil

# ================== 按你的路径修改这里 ==================
# 原始训练集路径
IMG_DIR = r"/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/dataset/6.2/images/train"
LABEL_DIR = r"/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/dataset/6.2/labels/train"

# 输出（只含 class2 的子集）
OUT_IMG_DIR = r"/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/dataset/6.2/images/train_class2_only"
OUT_LABEL_DIR = r"/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/dataset/6.2/labels/train_class2_only"

# 你要提取的类别 id（这里是第二类，class_id = 2）
TARGET_CLASS_ID = 2
# =======================================================

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

def label_contains_target(label_path, target_cls):
    """判断一个 label 文件里是否出现过指定类别"""
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls_id = int(parts[0])
            except (ValueError, IndexError):
                continue
            if cls_id == target_cls:
                return True
    return False

label_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))
print(f"共找到 {len(label_files)} 个 label 文件")

count = 0
miss_img = 0

for label_path in label_files:
    if not label_contains_target(label_path, TARGET_CLASS_ID):
        continue

    base = os.path.splitext(os.path.basename(label_path))[0]

    # 找对应的图片
    img_path = None
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        candidate = os.path.join(IMG_DIR, base + ext)
        if os.path.exists(candidate):
            img_path = candidate
            break

    if img_path is None:
        print(f"[警告] 找不到图片，对应标签: {label_path}")
        miss_img += 1
        continue

    # 复制图片 & 标签到新目录
    shutil.copy2(img_path, os.path.join(OUT_IMG_DIR, os.path.basename(img_path)))
    shutil.copy2(label_path, os.path.join(OUT_LABEL_DIR, os.path.basename(label_path)))

    count += 1

print("====================================")
print(f"含有 class {TARGET_CLASS_ID} 的样本数量：{count}")
print(f"其中找不到对应图片的标签文件数量：{miss_img}")
print("输出图片目录：", OUT_IMG_DIR)
print("输出标签目录：", OUT_LABEL_DIR)
