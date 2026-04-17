import sys
import torch
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径
sys.path.append("C:/Users/18084/Desktop/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main")

# 导入所有依赖模块
from ultralytics.nn.modules.block import C2PSA, PSABlock
from ultralytics.nn.extra_modules.transformer import TSSAlock_DYT_Mona_SEFN, DynamicTanh,C2TSSA_DYT_Mona_SEFN
from ultralytics.nn.extra_modules.semnet import SEFN
from ultralytics.nn.extra_modules.mona import Mona
from ultralytics.nn.extra_modules.attention import AttentionTSSA

# 创建模型
model = C2TSSA_DYT_Mona_SEFN(c1=64, c2=64, n=3, e=1)  # 确保 c2 >= 64

# 生成虚拟输入
dummy_input = torch.randn(1, 64, 224, 224)  # [batch, channels, height, width]

# 写入 TensorBoard
writer = SummaryWriter(log_dir="runs/model_visualization")
writer.add_graph(model, dummy_input)
writer.close()

print("可视化成功！")