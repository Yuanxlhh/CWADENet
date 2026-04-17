import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import numpy as np

# 加载训练好的模型
model = YOLO(r'C:\Users\18084\Desktop\yolo\YOLO\ultralytics-yolo11-20250502\runs\train\exp394\weights\best.pt')

# 创建一个字典来保存每个检测头（P3、P4、P5）的输出
feature_maps = {'p3': None, 'p4': None, 'p5': None}

# 定义 hook 函数来提取每个检测头的输出
def hook_fn(module, input, output):
    if isinstance(module, torch.nn.Conv2d):  # 只处理卷积层
        # 判断特征图的形状来确定检测头
        if output.shape[2] == 80 and output.shape[3] == 80:  # 可能是P3
            feature_maps['p3'] = output
        elif output.shape[2] == 40 and output.shape[3] == 40:  # 可能是P4
            feature_maps['p4'] = output
        elif output.shape[2] == 20 and output.shape[3] == 20:  # 可能是P5
            feature_maps['p5'] = output

# 注册 hook 到模型的 Detect_LSCSBD 模块
for name, module in model.model.named_modules():
    if isinstance(module, torch.nn.Conv2d):  # 仅为卷积层注册 hook
        module.register_forward_hook(hook_fn)

# 输入图像路径
img_path = r'C:\Users\18084\Desktop\yolo\YOLO\ultralytics-yolo11-20250502\ultralytics-yolo11-main\dataset\6.2\images\train\(4).jpg'

# 执行推理
results = model(img_path)  # 进行推理并保存结果

# 获取图片并绘制框
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

# 显示每个检测头的预测结果
def plot_predictions(feature_map, title, image):
    # 特征图后处理
    boxes = feature_map[..., :4]
    confidences = feature_map[..., 4:5]
    class_probs = feature_map[..., 5:]

    # 将框和预测的类别一起展示
    boxes = boxes.cpu().detach().numpy()
    confidences = confidences.cpu().detach().numpy()
    class_probs = class_probs.cpu().detach().numpy()

    # 用NMS去掉低置信度的框
    for i in range(boxes.shape[0]):
        for j in range(boxes.shape[1]):
            if confidences[i, j] > 0.5:  # 假设0.5为置信度阈值
                # 获取预测框坐标
                box = boxes[i, j]
                x1, y1, x2, y2 = box
                label = np.argmax(class_probs[i, j])  # 选择概率最大的类别

                # 绘制框和类别标签
                image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                image = cv2.putText(image, f'{label}: {confidences[i, j]:.2f}', (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# 显示 P3、P4 和 P5 的预测结果
if feature_maps['p3'] is not None:
    plot_predictions(feature_maps['p3'], "P3 Predictions", image.copy())
if feature_maps['p4'] is not None:
    plot_predictions(feature_maps['p4'], "P4 Predictions", image.copy())
if feature_maps['p5'] is not None:
    plot_predictions(feature_maps['p5'], "P5 Predictions", image.copy())
