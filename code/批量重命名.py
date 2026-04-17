# -*- coding: utf-8 -*-
from pathlib import Path

# ===== 改成你的图片文件夹路径 =====
IMG_DIR = Path(r"/home/xgq/Desktop/天气光照实验/雨天")

# ===== 起始编号 =====
START_NUM = 1

# ===== 支持的图片格式 =====
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    files = [p for p in IMG_DIR.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    files.sort()  # 按文件名排序，保证编号顺序稳定

    if not files:
        print("未找到图片，请检查路径是否正确。")
        return

    # 先改成临时名，避免重名冲突
    temp_files = []
    for i, f in enumerate(files):
        tmp_path = f.with_name(f"__tmp__{i}{f.suffix.lower()}")
        f.rename(tmp_path)
        temp_files.append(tmp_path)

    # 再改成最终名：400.png、401.png ...
    for i, f in enumerate(temp_files, start=START_NUM):
        new_name = f"{i}.png"
        new_path = f.with_name(new_name)
        f.rename(new_path)
        print(f"{f.name} -> {new_name}")

    print(f"\n完成，共重命名 {len(temp_files)} 张图片。")


if __name__ == "__main__":
    main()