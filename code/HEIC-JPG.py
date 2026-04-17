import pyheif
from PIL import Image
import os

def convert_heic_to_jpg(heic_file, output_dir=None):
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(heic_file)
    
    # 读取 HEIC 文件
    heif_file = pyheif.read(heic_file)
    
    # 转换为 PIL 图像对象
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data, 
        "raw", 
        heif_file.mode, 
        heif_file.stride,
    )
    
    # 获取文件名和输出路径
    file_name = os.path.basename(heic_file)
    output_name = os.path.splitext(file_name)[0] + ".jpg"
    output_path = os.path.join(output_dir, output_name)
    
    # 保存为 JPG 格式
    image.save(output_path, "JPEG")
    print(f"转换成功：{output_path}")

def convert_heic_folder_to_jpg(folder_path):
    # 遍历文件夹中的所有 HEIC 文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.heic'):
                heic_file_path = os.path.join(root, file)
                convert_heic_to_jpg(heic_file_path)

# 示例：将指定文件夹中的所有 HEIC 文件转换为 JPG 格式
heic_folder_path = "/home/xgq/Desktop/DJI/手机照片"  # 你的文件夹路径
convert_heic_folder_to_jpg(heic_folder_path)
