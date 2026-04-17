# import os
# import glob
# import cv2
# import albumentations as A

# # ================== 路径按你自己机器上的来 ==================
# SRC_IMG_DIR   = r"/home/xgq/Desktop/deadmood数据增强/images"
# SRC_LABEL_DIR = r"/home/xgq/Desktop/deadmood数据增强/labels"

# AUG_IMG_DIR   = r"/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/dataset/6.2/images/train_c2_aug"
# AUG_LABEL_DIR = r"/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/dataset/6.2/labels/train_c2_aug"
# # ========================================================

# os.makedirs(AUG_IMG_DIR, exist_ok=True)
# os.makedirs(AUG_LABEL_DIR, exist_ok=True)


# def clean_bbox(x, y, w, h):
#     """
#     把 YOLO bbox 限制在 [0,1]，并保证 x±w/2、y±h/2 不越界。
#     有问题就返回 None，把这个框丢掉。
#     """
#     # 先简单裁到 [0,1]
#     x = max(0.0, min(1.0, x))
#     y = max(0.0, min(1.0, y))
#     w = max(0.0, min(1.0, w))
#     h = max(0.0, min(1.0, h))

#     if w <= 0.0 or h <= 0.0:
#         return None

#     x_min = x - w / 2
#     x_max = x + w / 2
#     y_min = y - h / 2
#     y_max = y + h / 2

#     # 整个框都在画布外，直接丢弃
#     if x_max < 0 or x_min > 1 or y_max < 0 or y_min > 1:
#         return None

#     # 轻微越界就裁剪一下
#     x_min = max(0.0, x_min)
#     y_min = max(0.0, y_min)
#     x_max = min(1.0, x_max)
#     y_max = min(1.0, y_max)

#     w_new = x_max - x_min
#     h_new = y_max - y_min
#     if w_new <= 0.0 or h_new <= 0.0:
#         return None

#     x_new = (x_min + x_max) / 2.0
#     y_new = (y_min + y_max) / 2.0

#     # 再加一点点 margin，防止浮点误差导致刚好 >1
#     eps = 1e-6
#     x_new = max(0.0, min(1.0 - eps, x_new))
#     y_new = max(0.0, min(1.0 - eps, y_new))
#     w_new = max(0.0, min(1.0 - eps, w_new))
#     h_new = max(0.0, min(1.0 - eps, h_new))

#     return x_new, y_new, w_new, h_new


# def read_yolo_label(label_path):
#     """
#     读取 YOLO txt -> (bboxes, class_ids)，并做清洗
#     bbox 为归一化 [xc, yc, w, h]
#     """
#     bboxes = []
#     class_ids = []
#     with open(label_path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             parts = line.split()
#             if len(parts) < 5:
#                 continue
#             try:
#                 cls = int(parts[0])
#                 x, y, w, h = map(float, parts[1:5])
#             except ValueError:
#                 continue

#             cleaned = clean_bbox(x, y, w, h)
#             if cleaned is None:
#                 # 这个框太离谱，直接跳过
#                 continue
#             x_c, y_c, w_c, h_c = cleaned
#             bboxes.append([x_c, y_c, w_c, h_c])
#             class_ids.append(cls)
#     return bboxes, class_ids


# def save_yolo_label(label_path, bboxes, class_ids):
#     """保存 YOLO txt"""
#     with open(label_path, "w", encoding="utf-8") as f:
#         for cls, (x, y, w, h) in zip(class_ids, bboxes):
#             f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


# def post_clean_bboxes(bboxes, class_ids):
#     """
#     对增强后的 bbox 再做一次 clean_bbox，
#     丢掉明显越界/异常的框。
#     """
#     new_bboxes = []
#     new_classes = []
#     for cls, (x, y, w, h) in zip(class_ids, bboxes):
#         cleaned = clean_bbox(x, y, w, h)
#         if cleaned is None:
#             continue
#         new_bboxes.append(list(cleaned))
#         new_classes.append(cls)
#     return new_bboxes, new_classes


# # 统一的 bbox 配置
# BBOX_PARAMS = A.BboxParams(
#     format="yolo",
#     label_fields=["class_labels"],
#     min_visibility=0.3,
# )

# # ----------------- 1) 水平翻转 -----------------
# hflip_transform = A.Compose(
#     [A.HorizontalFlip(p=1.0)],
#     bbox_params=BBOX_PARAMS,
# )

# # ----------------- 2) 垂直翻转 -----------------
# vflip_transform = A.Compose(
#     [A.VerticalFlip(p=1.0)],
#     bbox_params=BBOX_PARAMS,
# )

# # ----------------- 3) 小角度旋转 -----------------
# def get_rotate_transform():
#     return A.Compose(
#         [
#             A.Rotate(
#                 limit=15,  # -15 ~ 15 度小角度
#                 border_mode=cv2.BORDER_CONSTANT,
#                 value=0,
#                 p=1.0,
#             )
#         ],
#         bbox_params=BBOX_PARAMS,
#     )

# # ----------------- 4) 仿射变换（含缩放+平移+轻微切变） -----------------
# def get_affine_transform():
#     return A.Compose(
#         [
#             A.Affine(
#                 scale=(0.9, 1.1),              # 轻微缩放
#                 translate_percent=(0.0, 0.1),  # 最多平移 10%
#                 rotate=(-10, 10),              # 再来一点小角度旋转
#                 shear=(-10, 10),               # 轻微切变
#                 fit_output=False,
#                 p=1.0,
#             )
#         ],
#         bbox_params=BBOX_PARAMS,
#     )

# # ----------------- 5) 随机裁剪 + 缩放 -----------------
# def get_random_crop_transform(h, w):
#     """
#     随机裁 80%~100% 区域，再缩放回原始大小，
#     实际效果相当于：随机裁剪 + 缩放（zoom）。
#     """
#     return A.Compose(
#         [
#             A.RandomResizedCrop(
#                 height=h,
#                 width=w,
#                 scale=(0.8, 1.0),
#                 ratio=(0.9, 1.1),
#                 p=1.0,
#             ),
#         ],
#         bbox_params=BBOX_PARAMS,
#     )

# # ----------------- 6) 颜色增强：亮度/对比度/饱和度 -----------------
# color_transform = A.Compose(
#     [
#         A.ColorJitter(
#             brightness=0.2,   # 亮度
#             contrast=0.2,     # 对比度
#             saturation=0.2,   # 饱和度
#             hue=0.0,          # 不改 hue
#             p=1.0,
#         )
#     ],
#     bbox_params=BBOX_PARAMS,
# )


# # ================== 主流程 ==================
# all_label_files = glob.glob(os.path.join(SRC_LABEL_DIR, "*.txt"))
# # 排除掉 classes.txt
# label_files = [p for p in all_label_files if os.path.basename(p) != "classes.txt"]

# print(f"共找到标签文件：{len(label_files)} 个")

# count_hflip = 0
# count_vflip = 0
# count_rot = 0
# count_affine = 0
# count_crop = 0
# count_color = 0

# for label_path in label_files:
#     base = os.path.splitext(os.path.basename(label_path))[0]

#     # 找图片（支持多种后缀）
#     img_path = None
#     for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]:
#         candidate = os.path.join(SRC_IMG_DIR, base + ext)
#         if os.path.exists(candidate):
#             img_path = candidate
#             break

#     if img_path is None:
#         print(f"[警告] 找不到图片: {base}")
#         continue

#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"[警告] 读图失败: {img_path}")
#         continue

#     h, w = img.shape[:2]
#     bboxes, class_ids = read_yolo_label(label_path)
#     if not bboxes:
#         # 这个文件可能所有框都被清洗掉了
#         continue

#     # -------- 1. 水平翻转 --------
#     try:
#         transformed = hflip_transform(
#             image=img,
#             bboxes=bboxes,
#             class_labels=class_ids,
#         )
#         if transformed["bboxes"]:
#             aug_bboxes, aug_class_ids = post_clean_bboxes(
#                 transformed["bboxes"], transformed["class_labels"]
#             )
#             if aug_bboxes:
#                 aug_img = transformed["image"]

#                 new_name = f"{base}_hflip"
#                 out_img_path = os.path.join(AUG_IMG_DIR, new_name + ".jpg")
#                 out_label_path = os.path.join(AUG_LABEL_DIR, new_name + ".txt")

#                 cv2.imwrite(out_img_path, aug_img)
#                 save_yolo_label(out_label_path, aug_bboxes, aug_class_ids)
#                 count_hflip += 1
#     except Exception as e:
#         print(f"[警告] {base} 水平翻转出错: {e}")

#     # -------- 2. 垂直翻转 --------
#     try:
#         transformed = vflip_transform(
#             image=img,
#             bboxes=bboxes,
#             class_labels=class_ids,
#         )
#         if transformed["bboxes"]:
#             aug_bboxes, aug_class_ids = post_clean_bboxes(
#                 transformed["bboxes"], transformed["class_labels"]
#             )
#             if aug_bboxes:
#                 aug_img = transformed["image"]

#                 new_name = f"{base}_vflip"
#                 out_img_path = os.path.join(AUG_IMG_DIR, new_name + ".jpg")
#                 out_label_path = os.path.join(AUG_LABEL_DIR, new_name + ".txt")

#                 cv2.imwrite(out_img_path, aug_img)
#                 save_yolo_label(out_label_path, aug_bboxes, aug_class_ids)
#                 count_vflip += 1
#     except Exception as e:
#         print(f"[警告] {base} 垂直翻转出错: {e}")

#     # -------- 3. 小角度旋转 --------
#     try:
#         rot_transform = get_rotate_transform()
#         transformed = rot_transform(
#             image=img,
#             bboxes=bboxes,
#             class_labels=class_ids,
#         )
#         if transformed["bboxes"]:
#             aug_bboxes, aug_class_ids = post_clean_bboxes(
#                 transformed["bboxes"], transformed["class_labels"]
#             )
#             if aug_bboxes:
#                 aug_img = transformed["image"]

#                 new_name = f"{base}_rot"
#                 out_img_path = os.path.join(AUG_IMG_DIR, new_name + ".jpg")
#                 out_label_path = os.path.join(AUG_LABEL_DIR, new_name + ".txt")

#                 cv2.imwrite(out_img_path, aug_img)
#                 save_yolo_label(out_label_path, aug_bboxes, aug_class_ids)
#                 count_rot += 1
#     except Exception as e:
#         print(f"[警告] {base} 旋转出错: {e}")

#     # -------- 4. 仿射变换（缩放+平移+切变） --------
#     try:
#         affine_transform = get_affine_transform()
#         transformed = affine_transform(
#             image=img,
#             bboxes=bboxes,
#             class_labels=class_ids,
#         )
#         if transformed["bboxes"]:
#             aug_bboxes, aug_class_ids = post_clean_bboxes(
#                 transformed["bboxes"], transformed["class_labels"]
#             )
#             if aug_bboxes:
#                 aug_img = transformed["image"]

#                 new_name = f"{base}_affine"
#                 out_img_path = os.path.join(AUG_IMG_DIR, new_name + ".jpg")
#                 out_label_path = os.path.join(AUG_LABEL_DIR, new_name + ".txt")

#                 cv2.imwrite(out_img_path, aug_img)
#                 save_yolo_label(out_label_path, aug_bboxes, aug_class_ids)
#                 count_affine += 1
#     except Exception as e:
#         print(f"[警告] {base} 仿射变换出错: {e}")

#     # -------- 5. 随机裁剪 + 缩放 --------
#     try:
#         crop_transform = get_random_crop_transform(h, w)
#         transformed = crop_transform(
#             image=img,
#             bboxes=bboxes,
#             class_labels=class_ids,
#         )
#         if transformed["bboxes"]:
#             aug_bboxes, aug_class_ids = post_clean_bboxes(
#                 transformed["bboxes"], transformed["class_labels"]
#             )
#             if aug_bboxes:
#                 aug_img = transformed["image"]

#                 new_name = f"{base}_crop"
#                 out_img_path = os.path.join(AUG_IMG_DIR, new_name + ".jpg")
#                 out_label_path = os.path.join(AUG_LABEL_DIR, new_name + ".txt")

#                 cv2.imwrite(out_img_path, aug_img)
#                 save_yolo_label(out_label_path, aug_bboxes, aug_class_ids)
#                 count_crop += 1
#     except Exception as e:
#         print(f"[警告] {base} 裁剪出错: {e}")

#     # -------- 6. 颜色增强（亮度/对比度/饱和度） --------
#     try:
#         transformed = color_transform(
#             image=img,
#             bboxes=bboxes,
#             class_labels=class_ids,
#         )
#         if transformed["bboxes"]:
#             aug_bboxes, aug_class_ids = post_clean_bboxes(
#                 transformed["bboxes"], transformed["class_labels"]
#             )
#             if aug_bboxes:
#                 aug_img = transformed["image"]

#                 new_name = f"{base}_color"
#                 out_img_path = os.path.join(AUG_IMG_DIR, new_name + ".jpg")
#                 out_label_path = os.path.join(AUG_LABEL_DIR, new_name + ".txt")

#                 cv2.imwrite(out_img_path, aug_img)
#                 save_yolo_label(out_label_path, aug_bboxes, aug_class_ids)
#                 count_color += 1
#     except Exception as e:
#         print(f"[警告] {base} 颜色增强出错: {e}")

# print("===================================")
# print(f"水平翻转样本数：{count_hflip}")
# print(f"垂直翻转样本数：{count_vflip}")
# print(f"小角度旋转样本数：{count_rot}")
# print(f"仿射变换样本数：{count_affine}")
# print(f"随机裁剪样本数：{count_crop}")
# print(f"颜色增强样本数：{count_color}")
# print("增强图片输出目录：", AUG_IMG_DIR)
# print("增强标签输出目录：", AUG_LABEL_DIR)
import os
import glob
import cv2
import albumentations as A

# ================== 路径按你自己的 crack 数据集来改 ==================
SRC_IMG_DIR   = r"/home/xgq/Desktop/裂缝数据集/images/train"
SRC_LABEL_DIR = r"/home/xgq/Desktop/裂缝数据集/labels/train"

AUG_IMG_DIR   = r"/home/xgq/Desktop/裂缝数据集/images/train_aug"
AUG_LABEL_DIR = r"/home/xgq/Desktop/裂缝数据集/labels/train_aug"
# ===================================================================

os.makedirs(AUG_IMG_DIR, exist_ok=True)
os.makedirs(AUG_LABEL_DIR, exist_ok=True)


def clean_bbox(x, y, w, h):
    """
    把 YOLO bbox 限制在 [0,1]，并保证 x±w/2、y±h/2 不越界。
    有问题就返回 None，把这个框丢掉。
    """
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    if w <= 0.0 or h <= 0.0:
        return None

    x_min = x - w / 2
    x_max = x + w / 2
    y_min = y - h / 2
    y_max = y + h / 2

    # 整个框都在画布外，直接丢弃
    if x_max < 0 or x_min > 1 or y_max < 0 or y_min > 1:
        return None

    # 裁剪到 [0, 1]
    x_min = max(0.0, x_min)
    y_min = max(0.0, y_min)
    x_max = min(1.0, x_max)
    y_max = min(1.0, y_max)

    w_new = x_max - x_min
    h_new = y_max - y_min
    if w_new <= 0.0 or h_new <= 0.0:
        return None

    x_new = (x_min + x_max) / 2.0
    y_new = (y_min + y_max) / 2.0

    eps = 1e-6
    x_new = max(0.0, min(1.0 - eps, x_new))
    y_new = max(0.0, min(1.0 - eps, y_new))
    w_new = max(0.0, min(1.0 - eps, w_new))
    h_new = max(0.0, min(1.0 - eps, h_new))

    return x_new, y_new, w_new, h_new


def read_yolo_label(label_path):
    """
    读取 YOLO txt -> (bboxes, class_ids)
    bbox 格式为归一化 [xc, yc, w, h]
    """
    bboxes = []
    class_ids = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
            except ValueError:
                continue

            cleaned = clean_bbox(x, y, w, h)
            if cleaned is None:
                continue

            x_c, y_c, w_c, h_c = cleaned
            bboxes.append([x_c, y_c, w_c, h_c])
            class_ids.append(cls)

    return bboxes, class_ids


def save_yolo_label(label_path, bboxes, class_ids):
    """保存 YOLO txt"""
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, (x, y, w, h) in zip(class_ids, bboxes):
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def post_clean_bboxes(bboxes, class_ids):
    """
    对增强后的 bbox 再做一次 clean_bbox，
    丢掉明显越界/异常的框。
    """
    new_bboxes = []
    new_classes = []

    for cls, (x, y, w, h) in zip(class_ids, bboxes):
        cleaned = clean_bbox(x, y, w, h)
        if cleaned is None:
            continue
        new_bboxes.append(list(cleaned))
        new_classes.append(cls)

    return new_bboxes, new_classes


# YOLO bbox 参数
BBOX_PARAMS = A.BboxParams(
    format="yolo",
    label_fields=["class_labels"],
    min_visibility=0.3,
)

# ----------------- 1) 水平翻转 -----------------
hflip_transform = A.Compose(
    [
        A.HorizontalFlip(p=1.0)
    ],
    bbox_params=BBOX_PARAMS,
)

# ----------------- 2) 垂直翻转 -----------------
vflip_transform = A.Compose(
    [
        A.VerticalFlip(p=1.0)
    ],
    bbox_params=BBOX_PARAMS,
)

# ----------------- 3) 小角度旋转 -----------------
def get_rotate_transform():
    return A.Compose(
        [
            A.Rotate(
                limit=15,  # -15 ~ 15 度
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0,
            )
        ],
        bbox_params=BBOX_PARAMS,
    )

# ----------------- 4) 仿射近似变换（兼容旧版 albumentations） -----------------
def get_affine_transform():
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.1,   # 平移最多 10%
                scale_limit=0.1,   # 缩放约 0.9~1.1
                rotate_limit=10,   # -10~10 度
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0,
            )
        ],
        bbox_params=BBOX_PARAMS,
    )

# ----------------- 5) 随机裁剪 + 缩放 -----------------
def get_random_crop_transform(h, w):
    return A.Compose(
        [
            A.RandomResizedCrop(
                height=h,
                width=w,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=1.0,
            )
        ],
        bbox_params=BBOX_PARAMS,
    )

# ----------------- 6) 颜色增强 -----------------
color_transform = A.Compose(
    [
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.0,
            p=1.0,
        )
    ],
    bbox_params=BBOX_PARAMS,
)

# ================== 主流程 ==================
all_label_files = glob.glob(os.path.join(SRC_LABEL_DIR, "*.txt"))
label_files = [p for p in all_label_files if os.path.basename(p) != "classes.txt"]

print(f"共找到标签文件：{len(label_files)} 个")

count_hflip = 0
count_vflip = 0
count_rot = 0
count_affine = 0
count_crop = 0
count_color = 0

for label_path in label_files:
    base = os.path.splitext(os.path.basename(label_path))[0]

    # 查找对应图片
    img_path = None
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]:
        candidate = os.path.join(SRC_IMG_DIR, base + ext)
        if os.path.exists(candidate):
            img_path = candidate
            break

    if img_path is None:
        print(f"[警告] 找不到图片: {base}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"[警告] 读图失败: {img_path}")
        continue

    h, w = img.shape[:2]
    bboxes, class_ids = read_yolo_label(label_path)

    if not bboxes:
        print(f"[提示] {base} 标签为空或清洗后无有效框，已跳过")
        continue

    # -------- 1. 水平翻转 --------
    try:
        transformed = hflip_transform(
            image=img,
            bboxes=bboxes,
            class_labels=class_ids
        )
        if transformed["bboxes"]:
            aug_bboxes, aug_class_ids = post_clean_bboxes(
                transformed["bboxes"], transformed["class_labels"]
            )
            if aug_bboxes:
                new_name = f"{base}_hflip"
                out_img_path = os.path.join(AUG_IMG_DIR, new_name + ".jpg")
                out_label_path = os.path.join(AUG_LABEL_DIR, new_name + ".txt")
                cv2.imwrite(out_img_path, transformed["image"])
                save_yolo_label(out_label_path, aug_bboxes, aug_class_ids)
                count_hflip += 1
    except Exception as e:
        print(f"[警告] {base} 水平翻转出错: {e}")

    # -------- 2. 垂直翻转 --------
    try:
        transformed = vflip_transform(
            image=img,
            bboxes=bboxes,
            class_labels=class_ids
        )
        if transformed["bboxes"]:
            aug_bboxes, aug_class_ids = post_clean_bboxes(
                transformed["bboxes"], transformed["class_labels"]
            )
            if aug_bboxes:
                new_name = f"{base}_vflip"
                out_img_path = os.path.join(AUG_IMG_DIR, new_name + ".jpg")
                out_label_path = os.path.join(AUG_LABEL_DIR, new_name + ".txt")
                cv2.imwrite(out_img_path, transformed["image"])
                save_yolo_label(out_label_path, aug_bboxes, aug_class_ids)
                count_vflip += 1
    except Exception as e:
        print(f"[警告] {base} 垂直翻转出错: {e}")

    # -------- 3. 小角度旋转 --------
    try:
        rot_transform = get_rotate_transform()
        transformed = rot_transform(
            image=img,
            bboxes=bboxes,
            class_labels=class_ids
        )
        if transformed["bboxes"]:
            aug_bboxes, aug_class_ids = post_clean_bboxes(
                transformed["bboxes"], transformed["class_labels"]
            )
            if aug_bboxes:
                new_name = f"{base}_rot"
                out_img_path = os.path.join(AUG_IMG_DIR, new_name + ".jpg")
                out_label_path = os.path.join(AUG_LABEL_DIR, new_name + ".txt")
                cv2.imwrite(out_img_path, transformed["image"])
                save_yolo_label(out_label_path, aug_bboxes, aug_class_ids)
                count_rot += 1
    except Exception as e:
        print(f"[警告] {base} 旋转出错: {e}")

    # -------- 4. ShiftScaleRotate（代替 Affine） --------
    try:
        affine_transform = get_affine_transform()
        transformed = affine_transform(
            image=img,
            bboxes=bboxes,
            class_labels=class_ids
        )
        if transformed["bboxes"]:
            aug_bboxes, aug_class_ids = post_clean_bboxes(
                transformed["bboxes"], transformed["class_labels"]
            )
            if aug_bboxes:
                new_name = f"{base}_affine"
                out_img_path = os.path.join(AUG_IMG_DIR, new_name + ".jpg")
                out_label_path = os.path.join(AUG_LABEL_DIR, new_name + ".txt")
                cv2.imwrite(out_img_path, transformed["image"])
                save_yolo_label(out_label_path, aug_bboxes, aug_class_ids)
                count_affine += 1
    except Exception as e:
        print(f"[警告] {base} 仿射近似变换出错: {e}")

    # -------- 5. 随机裁剪 + 缩放 --------
    try:
        crop_transform = get_random_crop_transform(h, w)
        transformed = crop_transform(
            image=img,
            bboxes=bboxes,
            class_labels=class_ids
        )
        if transformed["bboxes"]:
            aug_bboxes, aug_class_ids = post_clean_bboxes(
                transformed["bboxes"], transformed["class_labels"]
            )
            if aug_bboxes:
                new_name = f"{base}_crop"
                out_img_path = os.path.join(AUG_IMG_DIR, new_name + ".jpg")
                out_label_path = os.path.join(AUG_LABEL_DIR, new_name + ".txt")
                cv2.imwrite(out_img_path, transformed["image"])
                save_yolo_label(out_label_path, aug_bboxes, aug_class_ids)
                count_crop += 1
    except Exception as e:
        print(f"[警告] {base} 裁剪出错: {e}")

    # -------- 6. 颜色增强 --------
    try:
        transformed = color_transform(
            image=img,
            bboxes=bboxes,
            class_labels=class_ids
        )
        if transformed["bboxes"]:
            aug_bboxes, aug_class_ids = post_clean_bboxes(
                transformed["bboxes"], transformed["class_labels"]
            )
            if aug_bboxes:
                new_name = f"{base}_color"
                out_img_path = os.path.join(AUG_IMG_DIR, new_name + ".jpg")
                out_label_path = os.path.join(AUG_LABEL_DIR, new_name + ".txt")
                cv2.imwrite(out_img_path, transformed["image"])
                save_yolo_label(out_label_path, aug_bboxes, aug_class_ids)
                count_color += 1
    except Exception as e:
        print(f"[警告] {base} 颜色增强出错: {e}")

print("===================================")
print(f"水平翻转样本数：{count_hflip}")
print(f"垂直翻转样本数：{count_vflip}")
print(f"小角度旋转样本数：{count_rot}")
print(f"仿射近似变换样本数：{count_affine}")
print(f"随机裁剪样本数：{count_crop}")
print(f"颜色增强样本数：{count_color}")
print("增强图片输出目录：", AUG_IMG_DIR)
print("增强标签输出目录：", AUG_LABEL_DIR)