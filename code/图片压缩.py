# -*- coding: utf-8 -*-
import io
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

# ===== 按你的路径与输出位置设置 =====
SRC = Path(r"/home/xgq/Desktop/天气光照实验/夜间1/images")
# 在输入目录下创建 compressed 文件夹
DST = SRC.parent / "visible_images_compressed"
MAX_SIDE = 1280      # imgsz=640 推荐 1280；更省可改 1024
JPEG_QUALITY = 82    # 经验“甜点位”
KEEP_DIRS = True     # 保留子目录结构
FLATTEN_ALPHA = True # PNG 有透明就铺白后转 JPG（适合训练）

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def calc_size(w, h, max_side):
    if max(w, h) <= max_side:
        return w, h
    s = max_side / float(max(w, h))
    return int(w * s + 0.5), int(h * s + 0.5)

def jpg_bytes(im: Image.Image, quality=82, flatten_alpha=True) -> bytes:
    # 透明转白底；非 RGB 统一转 RGB
    if "A" in im.getbands() and flatten_alpha:
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    elif im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True, subsampling="4:2:0")
    return buf.getvalue()

def main():
    assert SRC.exists(), f"输入目录不存在：{SRC}"
    DST.mkdir(parents=True, exist_ok=True)
    files = [p for p in SRC.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not files:
        print("未找到图片。"); return

    total_in, total_out = 0, 0
    for src in tqdm(files, ncols=90, desc="Compressing"):
        rel = src.relative_to(SRC) if KEEP_DIRS else Path(src.name)
        dst = (DST / rel).with_suffix(".jpg")
        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            im = Image.open(src)
            im = ImageOps.exif_transpose(im)   # 应用 EXIF 方向
            w, h = im.size
            nw, nh = calc_size(w, h, MAX_SIDE)
            if (nw, nh) != (w, h):
                im = im.resize((nw, nh), Image.LANCZOS)
            out = jpg_bytes(im, JPEG_QUALITY, FLATTEN_ALPHA)
            dst.write_bytes(out)
            out_size = len(out)
        except Exception:
            # 读写异常就把原图复制过去，保证不中断
            data = src.read_bytes()
            dst.write_bytes(data)
            out_size = len(data)

        total_in  += src.stat().st_size
        total_out += out_size

    gb = 1024**3
    print("\n=== Summary ===")
    print(f"Files: {len(files)}")
    print(f"Total before: {total_in/gb:.2f} GB")
    print(f"Total after : {total_out/gb:.2f} GB")
    if total_in:
        print(f"Reduced     : {(1-total_out/total_in)*100:.1f}%")

if __name__ == "__main__":
    main()
