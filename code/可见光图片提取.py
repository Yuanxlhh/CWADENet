# -*- coding: utf-8 -*-
from pathlib import Path
import shutil

# =========================
# 1. 总数据目录（这里改成你的总文件夹路径）
# =========================
SRC_DIR = Path(r"/home/xgq/Desktop/3.21石头城无人机数据")

# =========================
# 2. 提取后的保存目录
# =========================
DST_DIR = Path(r"/home/xgq/Desktop/visible_images")

def is_visible_image(file_path: Path) -> bool:
    """
    只提取可见光图片：
    文件名一般以 _V.JPG 结尾
    例如：
    DJI_20260321091823_0013_V.JPG
    """
    return (
        file_path.is_file()
        and file_path.suffix.lower() in [".jpg", ".jpeg"]
        and file_path.stem.upper().endswith("_V")
    )

def get_unique_save_path(dst_dir: Path, filename: str) -> Path:
    """
    如果目标文件夹里出现重名文件，自动重命名
    例如：
    a.jpg -> a_1.jpg -> a_2.jpg
    """
    save_path = dst_dir / filename
    if not save_path.exists():
        return save_path

    stem = save_path.stem
    suffix = save_path.suffix
    idx = 1
    while True:
        new_path = dst_dir / f"{stem}_{idx}{suffix}"
        if not new_path.exists():
            return new_path
        idx += 1

def main():
    if not SRC_DIR.exists():
        print(f"源目录不存在：{SRC_DIR}")
        return

    DST_DIR.mkdir(parents=True, exist_ok=True)

    copied_count = 0

    # 递归遍历所有子文件夹
    for file_path in SRC_DIR.rglob("*"):
        if not is_visible_image(file_path):
            continue

        save_path = get_unique_save_path(DST_DIR, file_path.name)
        shutil.copy2(file_path, save_path)
        copied_count += 1
        print(f"[{copied_count}] 已复制: {file_path.name}")

    print("\n✅ 提取完成")
    print(f"共提取可见光图片数量：{copied_count}")
    print(f"保存位置：{DST_DIR.resolve()}")

if __name__ == "__main__":
    main()
