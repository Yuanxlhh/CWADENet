import os
from collections import Counter

# 改成你自己的训练集 labels 目录
LABEL_DIR = r"/home/xgq/Desktop/裂缝数据集/labels/1"

cls_counter = Counter()
file_counter = 0
empty_files = 0

for root, _, files in os.walk(LABEL_DIR):
    for name in files:
        if not name.lower().endswith(".txt"):
            continue
        file_counter += 1
        path = os.path.join(root, name)

        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if not lines:
            empty_files += 1
            continue

        for line in lines:
            parts = line.split()
            try:
                cls_id = int(parts[0])   # YOLO 的第一列是类别 id
            except (ValueError, IndexError):
                continue
            cls_counter[cls_id] += 1

total_labels = sum(cls_counter.values())

print("====== YOLO 训练集标注统计 ======")
print(f"标注文件数量（有 txt 的图）：{file_counter}")
print(f"其中空标注文件（0 个框）： {empty_files}")
print(f"总标注框数量：{total_labels}\n")

print("按类别统计（class_id : 数量）：")
for cls_id in sorted(cls_counter.keys()):
    print(f"  {cls_id}: {cls_counter[cls_id]}")

print("\n如果你有 classes.txt，可以自己对照 class_id 看类别名称。")
