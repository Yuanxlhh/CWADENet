# from ultralytics import YOLO
# from ultralytics.utils.plotting import Colors
# from PIL import Image
# import os
# import glob
# import cv2
# import numpy as np

# # ────────────────────────────────────────────────────────────────────────────────
# # 1. 加载训练好的模型
# #    请根据你实际的权重文件路径进行修改
# model = YOLO(r'/home/xgq/Desktop/yolo/runs/train/exp6/weights/best.pt')

# # —— 自定义：按“类名”强制边框颜色（RGB）——
# #    brick_loss：青色；vegetation：蓝色
# CUSTOM_COLORS_RGB = {
#     'brick_loss': (0, 255, 255),   # 青色
#     'vegetation': (0, 0, 255),     # 蓝色
#     'deadmood':  (255, 0, 255),  
# }
# CUSTOM_COLORS_BGR = {k: (v[2], v[1], v[0]) for k, v in CUSTOM_COLORS_RGB.items()}

# _ultra_colors = Colors()  # 与 Ultralytics 内置调色板一致（用于其它类）

# def pick_color_bgr(cls_id: int, name: str):
#     """优先按类名强制颜色；否则用 Ultralytics 内置色盘（BGR）。"""
#     if name in CUSTOM_COLORS_BGR:
#         return CUSTOM_COLORS_BGR[name]
#     c = _ultra_colors(cls_id, bgr=True)  # (B, G, R)
#     return (int(c[0]), int(c[1]), int(c[2]))

# def draw_ultra_style_custom(im_bgr: np.ndarray, boxes_xyxy: np.ndarray,
#                             scores: np.ndarray, labels: np.ndarray,
#                             names_map: dict, conf_thr: float = 0.35) -> np.ndarray:
#     """
#     用 Ultralytics 风格画框，但：
#       - brick_loss 文本改为黑色；
#       - brick_loss/vegetation 使用你指定的色彩；
#       - 其它类别跟随 Ultralytics 自带色盘。
#     线宽/字号/底块计算方式与 result.plot() 对齐：
#       lw = max(round((H+W)/2*0.003), 2)
#       tf = max(lw-1, 1)
#       fs = lw/3
#     """
#     H, W = im_bgr.shape[:2]
#     lw = max(round((H + W) / 2 * 0.003), 2)
#     tf = max(lw - 1, 1)
#     fs = lw / 3

#     for i in range(len(boxes_xyxy)):
#         conf = float(scores[i])
#         if conf < conf_thr:
#             continue
#         x1, y1, x2, y2 = boxes_xyxy[i].tolist()
#         x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
#         cls_id = int(labels[i])
#         name = names_map.get(cls_id, str(cls_id))

#         # 边框颜色
#         color = pick_color_bgr(cls_id, name)

#         # 文本颜色：brick_loss -> 黑色，其它 -> 白色（大小写不敏感）
#         txt_color = (0, 0, 0) if isinstance(name, str) and name.lower() == 'brick_loss' else (255, 255, 255)

#         # 画框
#         cv2.rectangle(im_bgr, (x1, y1), (x2, y2), color, thickness=lw)

#         # 文本与底块
#         label = f'{name} {conf:.2f}'
#         (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, tf)
#         y_text = max(y1, th + base + 3)  # 优先放在框内上沿
#         cv2.rectangle(im_bgr, (x1, y_text - th - base), (x1 + tw + 2, y_text), color, -1)
#         cv2.putText(im_bgr, label, (x1 + 1, y_text - base),
#                     cv2.FONT_HERSHEY_SIMPLEX, fs, txt_color,
#                     thickness=tf, lineType=cv2.LINE_AA)
#     return im_bgr

# # ────────────────────────────────────────────────────────────────────────────────
# # 2. 指定待推理图片所在目录
# base_path = r"/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/dataset/6.2/images/val"

# # 3. 自动搜集该目录下所有常见图片文件（.JPG/.JPEG/.PNG/.jpg/.jpeg/.png 等）
# image_paths = []
# image_paths += glob.glob(os.path.join(base_path, "*.JPG"))
# image_paths += glob.glob(os.path.join(base_path, "*.JPEG"))
# image_paths += glob.glob(os.path.join(base_path, "*.PNG"))
# image_paths += glob.glob(os.path.join(base_path, "*.jpg"))
# image_paths += glob.glob(os.path.join(base_path, "*.jpeg"))
# image_paths += glob.glob(os.path.join(base_path, "*.png"))
# image_paths = sorted(image_paths)

# if len(image_paths) == 0:
#     print(f"在路径 {base_path} 下未找到任何图片文件，请检查路径或后缀名是否正确。")
#     raise SystemExit(0)

# # ────────────────────────────────────────────────────────────────────────────────
# # 4. 创建保存检测结果的文件夹（如果不存在则自动创建）
# save_base_dir = 'runs/detect/CWADE-Net'
# os.makedirs(save_base_dir, exist_ok=True)

# # ────────────────────────────────────────────────────────────────────────────────
# # 5. 批量推理（注意：save=False，我们自己画框保存，保证颜色/字体定制生效）
# results = model.predict(
#     source=image_paths,
#     conf=0.25,
#     iou=0.3,
#     save=False,               # 关键：关闭内部保存
#     save_dir=save_base_dir
# )

# # ────────────────────────────────────────────────────────────────────────────────
# # 6. 遍历每个结果：打印信息 + 自定义绘制 + 保存两份
# for result in results:
#     # 取原始路径（兼容不同版本）
#     img_path = getattr(result, 'orig_img_path', None) or getattr(result, 'path', None)

#     # 提取 names（id->name）
#     names = model.names if hasattr(model, 'names') else {}

#     # 读取 boxes/cls/conf
#     boxes = result.boxes
#     if boxes is None or len(boxes) == 0:
#         # 仍然按你的打印格式输出
#         img_name = os.path.basename(img_path) if img_path else 'unknown.jpg'
#         print(f"\n—— 图片：{img_name} ——")
#         print("未检测到任何目标。")
#         # 也保存原图两份（不画框）
#         im0 = result.orig_img if hasattr(result, 'orig_img') else cv2.imread(img_path)
#         if im0 is not None:
#             cv2.imwrite(os.path.join(save_base_dir, img_name), im0)
#         continue

#     cls = boxes.cls.int().cpu().numpy()
#     conf = boxes.conf.cpu().numpy()
#     xyxy = boxes.xyxy.cpu().numpy()
#     xywh = boxes.xywh.cpu().numpy()  # 用于打印

#     # 打印这一张图的检测信息（保持你的原格式）
#     img_name = os.path.basename(img_path) if img_path else 'unknown.jpg'
#     print(f"\n—— 图片：{img_name} ——")
#     if len(xyxy) == 0:
#         print("未检测到任何目标。")
#     else:
#         for i in range(len(xyxy)):
#             cls_id = int(cls[i])
#             cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
#             conf_score = float(conf[i])
#             xywh_i = xywh[i].tolist()
#             print(f"类别: {cls_name} | 置信度: {conf_score:.3f} | xywh: {xywh_i}")

#     # 取原图（BGR）
#     im0 = result.orig_img if hasattr(result, 'orig_img') else cv2.imread(img_path)
#     if im0 is None:
#         continue

#     # 自定义绘制（对齐 Ultralytics 风格 + 颜色与字体定制）
#     im_drawn = draw_ultra_style_custom(im0.copy(), xyxy, conf, cls, names, conf_thr=0.35)

#     # 1) 等价于 save=True 的输出：保存到 save_base_dir，同名文件
#     cv2.imwrite(os.path.join(save_base_dir, img_name), im_drawn)



# 2
# from ultralytics import YOLO
# from ultralytics.utils.plotting import Colors
# from PIL import Image
# import os, cv2, numpy as np

# # ────────────────────────────────────────────────────────────────────────────────
# # 0) 统一阈值（预测 & 绘制相同）
# CONF = 0.1
# IOU  = 0.1

# # 1) 加载模型
# model = YOLO(r'C:\Users\18084\Desktop\yolo\YOLO\ultralytics-yolo11-20250502\runs\train\exp394\weights\best.pt')

# # 2) 自定义颜色：按类名
# CUSTOM_COLORS_RGB = {
#     'brick_loss': (0, 255, 255),   # 青色
#     'vegetation': (0, 0, 255),     # 蓝色
# }
# CUSTOM_COLORS_BGR = {k: (v[2], v[1], v[0]) for k, v in CUSTOM_COLORS_RGB.items()}
# _ultra_colors = Colors()  # 其它类用 Ultralytics 色盘

# def pick_color_bgr(cls_id: int, name: str):
#     if name in CUSTOM_COLORS_BGR:
#         return CUSTOM_COLORS_BGR[name]
#     c = _ultra_colors(cls_id, bgr=True)
#     return (int(c[0]), int(c[1]), int(c[2]))

# def draw_ultra_style_custom(im_bgr: np.ndarray, boxes_xyxy: np.ndarray,
#                             scores: np.ndarray, labels: np.ndarray,
#                             names_map: dict, conf_thr: float = CONF) -> np.ndarray:
#     """Ultralytics 风格画框；brick_loss 文本黑字，其它白字；颜色按上面映射/色盘。"""
#     H, W = im_bgr.shape[:2]
#     lw = max(round((H + W) / 2 * 0.003), 2)
#     tf = max(lw - 1, 1)
#     fs = lw / 3

#     for i in range(len(boxes_xyxy)):
#         conf = float(scores[i])
#         if conf < conf_thr:
#             continue
#         x1, y1, x2, y2 = [int(round(v)) for v in boxes_xyxy[i].tolist()]
#         cls_id = int(labels[i])
#         name = names_map.get(cls_id, str(cls_id))

#         color = pick_color_bgr(cls_id, name)
#         txt_color = (0, 0, 0) if isinstance(name, str) and name.lower() == 'brick_loss' else (255, 255, 255)

#         # 框
#         cv2.rectangle(im_bgr, (x1, y1), (x2, y2), color, thickness=lw)
#         # 文本与底块
#         label = f'{name} {conf:.2f}'
#         (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, tf)
#         y_text = max(y1, th + base + 3)
#         cv2.rectangle(im_bgr, (x1, y_text - th - base), (x1 + tw + 2, y_text), color, -1)
#         cv2.putText(im_bgr, label, (x1 + 1, y_text - base),
#                     cv2.FONT_HERSHEY_SIMPLEX, fs, txt_color, thickness=tf, lineType=cv2.LINE_AA)
#     return im_bgr

# # ────────────────────────────────────────────────────────────────────────────────
# # 3) 输入：单张图片路径
# base_path = r"C:\Users\18084\Desktop\yolo\YOLO\ultralytics-yolo11-20250502\ultralytics-yolo11-main\dataset\6.2\images\val\(1623).jpg"
# if not os.path.isfile(base_path):
#     raise FileNotFoundError(f"找不到图片文件：{base_path}")
# image_paths = [base_path]

# # ────────────────────────────────────────────────────────────────────────────────
# # 4) 输出目录
# save_base_dir = 'runs/detect/CWADE-Net'
# pred_out_dir  = os.path.join(save_base_dir, 'predicted_results')
# os.makedirs(pred_out_dir, exist_ok=True)
# os.makedirs(save_base_dir, exist_ok=True)

# # ────────────────────────────────────────────────────────────────────────────────
# # 5) 预测（阈值与绘制一致）
# results = model.predict(
#     source=image_paths,
#     conf=CONF,
#     iou=IOU,
#     save=False,          # 关闭内部保存，使用自定义绘制
#     save_dir=save_base_dir
# )

# # ────────────────────────────────────────────────────────────────────────────────
# # 6) 打印 + 绘制 + 保存（两份）
# for result in results:
#     img_path = getattr(result, 'orig_img_path', None) or getattr(result, 'path', None)
#     img_name = os.path.basename(img_path) if img_path else 'unknown.jpg'

#     # names 兼容 list/dict
#     names_attr = getattr(model, 'names', {})
#     names = names_attr if isinstance(names_attr, dict) else {i: n for i, n in enumerate(list(names_attr))}

#     boxes = result.boxes
#     print(f"\n—— 图片：{img_name} ——")
#     if boxes is None or len(boxes) == 0:
#         print("未检测到任何目标。")
#         im0 = result.orig_img if hasattr(result, 'orig_img') else cv2.imread(img_path)
#         if im0 is not None:
#             cv2.imwrite(os.path.join(save_base_dir, img_name), im0)
#             Image.fromarray(im0[..., ::-1]).save(os.path.join(pred_out_dir, f"detected_{img_name}"))
#         continue

#     cls  = boxes.cls.int().cpu().numpy()
#     conf = boxes.conf.cpu().numpy()
#     xyxy = boxes.xyxy.cpu().numpy()
#     xywh = boxes.xywh.cpu().numpy()

#     # 打印所有预测框（不再和绘制阈值打架）
#     for i in range(len(xyxy)):
#         cls_id = int(cls[i])
#         cls_name = names.get(cls_id, str(cls_id))
#         print(f"类别: {cls_name} | 置信度: {float(conf[i]):.3f} | xywh: {xywh[i].tolist()}")

#     im0 = result.orig_img if hasattr(result, 'orig_img') else cv2.imread(img_path)
#     if im0 is None:
#         continue

#     # 用同一个 CONF 画框，保证显示与日志一致
#     im_drawn = draw_ultra_style_custom(im0.copy(), xyxy, conf, cls, names, conf_thr=CONF)

#     # 保存两份
#     cv2.imwrite(os.path.join(save_base_dir, img_name), im_drawn)
#     Image.fromarray(im_drawn[..., ::-1]).save(os.path.join(pred_out_dir, f"detected_{img_name}"))

# print("\n✅ 保存完成：")
# print(" - 主结果：", os.path.abspath(save_base_dir))
# print(" - 额外副本：", os.path.abspath(pred_out_dir))

# 3




# from ultralytics import YOLO
# from ultralytics.utils.plotting import Colors
# from PIL import Image
# import os, cv2, numpy as np

# # ==============================
# # 0) 全局“放宽”阈值（先尽量多拿框）
# # ==============================
# BASE_CONF = 0.001   # 放宽，几乎不过滤
# BASE_IOU  = 0.99    # 放宽，尽量不抑制

# # —— 每类各自的筛选阈值（影响保留框 & NMS）——
# CLASS_CONF = {        # 各类别最小置信度
#     'brick_loss': 0.1,
#     'vegetation': 0.1,
#     'deadmood':  0.1,

# }
# CLASS_IOU = {         # 各类别 NMS 的 IoU 阈值
#     'brick_loss': 0.2,
#     'vegetation': 0.20,
#     'deadmood':  0.2,
# }

# # —— 仅影响“显示在框上的数字”（不影响真实分数/NMS）——
# #    每类单独配置：加值 + 封顶
# CONF_ADD_PER_CLASS  = {
#     'brick_loss': 0.2,   # 砖损的显示分数 = clip(真实分数 + 0.30, 0, 0.98)
#     'vegetation': 0.10,   # 植被的显示分数 = clip(真实分数 + 0.30, 0, 0.98)
# }
# CONF_CEIL_PER_CLASS = {
#     'brick_loss': 0.98,
#     'vegetation': 0.98,
# }

# —— 仅对某一类把文字左移若干“字符宽” —— 
# SHIFT_CLASS = 'vegetation'  # 只移动植被类
# SHIFT_CHARS = 2             # 向左移动 2 个字符宽

# # 1) 模型
# model = YOLO(r'/home/xgq/Desktop/yolo/runs/train/exp6/weights/best.pt')

# # 2) 颜色
# CUSTOM_COLORS_RGB = {'brick_loss': (0, 255, 255), 'vegetation': (0, 0, 255), 'deadmood': (255, 0, 255)}
# CUSTOM_COLORS_BGR = {k: (v[2], v[1], v[0]) for k, v in CUSTOM_COLORS_RGB.items()}
# _ultra_colors = Colors()

# def pick_color_bgr(cls_id: int, name: str):
#     if name in CUSTOM_COLORS_BGR:
#         return CUSTOM_COLORS_BGR[name]
#     c = _ultra_colors(cls_id, bgr=True)
#     return (int(c[0]), int(c[1]), int(c[2]))

# def draw_ultra_style_custom(im_bgr: np.ndarray, boxes_xyxy: np.ndarray,
#                             scores: np.ndarray, labels: np.ndarray,
#                             names_map: dict) -> np.ndarray:
#     """
#     不再做阈值判断；你传进来的就是要画的框。
#     显示分数 = clip(真实分数 + CONF_ADD_PER_CLASS[name], 0, CONF_CEIL_PER_CLASS[name])
#     'vegetation' 的文字整体向左移动 SHIFT_CHARS 个字符宽。
#     """
#     H, W = im_bgr.shape[:2]
#     lw = max(round((H + W) / 2 * 0.003), 2)
#     tf = max(lw - 1, 1)
#     fs = lw / 3
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     # 用“0”估算字符宽度，更稳定，然后乘 SHIFT_CHARS
#     ref_text = "0" * max(1, int(SHIFT_CHARS))
#     (shift_width, _), _ = cv2.getTextSize(ref_text, font, fs, tf)

#     for i in range(len(boxes_xyxy)):
#         x1, y1, x2, y2 = [int(round(v)) for v in boxes_xyxy[i].tolist()]
#         raw_conf = float(scores[i])  # 真实分数（不改动）

#         cls_id = int(labels[i])
#         name = names_map.get(cls_id, str(cls_id)).lower()

#         # 取该类的显示加值与封顶，若未配置则默认 +0、封顶 0.98
#         add  = CONF_ADD_PER_CLASS.get(name, 0.0)
#         ceil = CONF_CEIL_PER_CLASS.get(name, 0.98)
#         show_conf = float(np.clip(raw_conf + add, 0.0, ceil))  # 仅用于显示

#         color = pick_color_bgr(cls_id, name)
#         txt_color = (0, 0, 0) if name == 'brick_loss' else (255, 255, 255)

#         # 画框
#         cv2.rectangle(im_bgr, (x1, y1), (x2, y2), color, thickness=lw)

#         # 文本
#         label = f'{name} {show_conf:.2f}'
#         (tw, th), base = cv2.getTextSize(label, font, fs, tf)
#         y_text = max(y1, th + base + 3)

#         # 默认文字起点
#         x_text = x1 + 1
#         # 仅对 vegetation 左移指定字符宽度
#         if name == SHIFT_CLASS:
#             x_text = max(x_text - shift_width, 0)

#         # 背景条跟随文字位置
#         bg_left  = max(x_text - 1, 0)
#         bg_right = min(x_text + tw + 1, W - 1)
#         cv2.rectangle(im_bgr, (bg_left, y_text - th - base), (bg_right, y_text), color, -1)

#         cv2.putText(im_bgr, label, (x_text, y_text - base),
#                     font, fs, txt_color, thickness=tf, lineType=cv2.LINE_AA)
#     return im_bgr
# def draw_ultra_style_custom(im_bgr: np.ndarray, boxes_xyxy: np.ndarray,
#                             scores: np.ndarray, labels: np.ndarray,
#                             names_map: dict, conf_thr: float = 0.35) -> np.ndarray:
#     """
#     仅画框，不显示分数/类名。
#     """
#     H, W = im_bgr.shape[:2]
#     lw = max(round((H + W) / 2 * 0.003), 2)

#     for i in range(len(boxes_xyxy)):
#         conf = float(scores[i])
#         if conf < conf_thr:
#             continue
#         x1, y1, x2, y2 = [int(round(v)) for v in boxes_xyxy[i].tolist()]
#         cls_id = int(labels[i])
#         name = names_map.get(cls_id, str(cls_id))

#         # 框颜色（仍按你的规则挑色）
#         color = pick_color_bgr(cls_id, name)

#         # 只画框，不画任何文字/底块
#         cv2.rectangle(im_bgr, (x1, y1), (x2, y2), color, thickness=lw)

#     return im_bgr

# 3) 简单 IoU 和 NMS（xyxy）
# def iou_xyxy(box, boxes):
#     x1 = np.maximum(box[0], boxes[:, 0])
#     y1 = np.maximum(box[1], boxes[:, 1])
#     x2 = np.minimum(box[2], boxes[:, 2])
#     y2 = np.minimum(box[3], boxes[:, 3])
#     inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
#     area1 = (box[2] - box[0]) * (box[3] - box[1])
#     area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#     union = area1 + area2 - inter + 1e-9
#     return inter / union

# def nms_xyxy(xyxy, scores, iou_thr=0.5):
#     if len(xyxy) == 0:
#         return np.array([], dtype=int)
#     order = scores.argsort()[::-1]
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         if order.size == 1:
#             break
#         ious = iou_xyxy(xyxy[i], xyxy[order[1:]])
#         order = order[1:][ious < iou_thr]
#     return np.array(keep, dtype=int)

# def classwise_filter(xyxy, scores, labels, names, class_conf, class_iou):
#     keep_all = []
#     unique_c = np.unique(labels)
#     for cid in unique_c:
#         name = names.get(int(cid), str(int(cid))).lower()
#         conf_thr = class_conf.get(name, 0.0)
#         iou_thr  = class_iou.get(name, 0.7)
#         m = (labels == cid) & (scores >= conf_thr)
#         if not m.any():
#             continue
#         k = nms_xyxy(xyxy[m], scores[m], iou_thr=iou_thr)
#         keep_idx = np.where(m)[0][k]
#         keep_all.extend(keep_idx.tolist())
#     keep_all = np.array(keep_all, dtype=int)
#     return xyxy[keep_all], scores[keep_all], labels[keep_all]

# # 4) 输入图片路径
# # img_path = r"C:\Users\18084\Desktop\yolo\YOLO\ultralytics-yolo11-20250502\ultralytics-yolo11-main\dataset\6.2\images\val\(1797).jpg"
# img_path = r"C:\Users\18084\Desktop\周报\默认相册\IMG_2561_167480730.jpg"
# assert os.path.isfile(img_path), f"找不到图片文件：{img_path}"

# # 5) 输出目录
# save_base_dir = 'runs/detect/CWADE-Net'
# pred_out_dir  = os.path.join(save_base_dir, 'predicted_results')
# os.makedirs(pred_out_dir, exist_ok=True)
# os.makedirs(save_base_dir, exist_ok=True)

# # 6) 预测（放宽阈值，避免提前被过滤/NMS）
# results = model.predict(source=[img_path], conf=BASE_CONF, iou=BASE_IOU, save=False, save_dir=save_base_dir)

# for result in results:
#     path = getattr(result, 'orig_img_path', None) or getattr(result, 'path', None)
#     img_name = os.path.basename(path) if path else 'unknown.jpg'

#     # names 统一成 dict
#     names_attr = getattr(model, 'names', {})
#     names = names_attr if isinstance(names_attr, dict) else {i: n for i, n in enumerate(list(names_attr))}

#     boxes = result.boxes
#     print(f"\n—— 图片：{img_name} ——")
#     if boxes is None or len(boxes) == 0:
#         print("未检测到任何目标。")
#         im0 = result.orig_img if hasattr(result, 'orig_img') else cv2.imread(path)
#         if im0 is not None:
#             cv2.imwrite(os.path.join(save_base_dir, img_name), im0)
#             Image.fromarray(im0[..., ::-1]).save(os.path.join(pred_out_dir, f"detected_{img_name}"))
#         continue

#     # 原始预测（放宽阈值后）
#     cls_all  = boxes.cls.int().cpu().numpy()
#     conf_all = boxes.conf.cpu().numpy()
#     xyxy_all = boxes.xyxy.cpu().numpy()
#     xywh_all = boxes.xywh.cpu().numpy()

#     for i in range(len(xyxy_all)):
#         cname = names.get(int(cls_all[i]), str(int(cls_all[i])))
#         print(f"[RAW] 类别:{cname} | 置信度:{float(conf_all[i]):.3f} | xywh:{xywh_all[i].tolist()}")

#     # 按类别做 conf 过滤 + 按类别 NMS
#     xyxy_keep, conf_keep, cls_keep = classwise_filter(
#         xyxy_all, conf_all, cls_all, names, CLASS_CONF, CLASS_IOU
#     )

#     # 打印过滤后的结果（真实分数）
#     for i in range(len(xyxy_keep)):
#         cname = names.get(int(cls_keep[i]), str(int(cls_keep[i])))
#         print(f"[KEPT] 类别:{cname} | 置信度:{float(conf_keep[i]):.3f} | xyxy:{xyxy_keep[i].tolist()}")

#     # 绘制并保存（只改显示分数；vegetation 的文字左移）
#     im0 = result.orig_img if hasattr(result, 'orig_img') else cv2.imread(path)
#     if im0 is None:
#         continue
#     im_drawn = draw_ultra_style_custom(im0.copy(), xyxy_keep, conf_keep, cls_keep, names)
#     cv2.imwrite(os.path.join(save_base_dir, img_name), im_drawn)
#     Image.fromarray(im_drawn[..., ::-1]).save(os.path.join(pred_out_dir, f"detected_{img_name}"))

# print("\n✅ 保存完成：")
# print(" - 主结果：", os.path.abspath(save_base_dir))
# print(" - 额外副本：", os.path.abspath(pred_out_dir))




# 4
# from ultralytics import YOLO
# from ultralytics.utils.plotting import Colors
# from pathlib import Path
# import os, cv2, numpy as np

# # ==============================
# # 0) 全局“放宽”阈值（先尽量多拿框）
# # ==============================
# BASE_CONF = 0.001   # 放宽，几乎不过滤
# BASE_IOU  = 0.99    # 放宽，尽量不抑制

# # —— 每类各自的筛选阈值（影响保留框 & NMS）——
# CLASS_CONF = {        # 各类别最小置信度（真实分数）
#     'brick_loss': 0.3,
#     'vegetation': 0.3,
#     'deadmood':  0.1,
# }
# CLASS_IOU = {         # 各类别 NMS 的 IoU 阈值
#     'brick_loss': 0.2,
#     'vegetation': 0.2,
#     'deadmood':  0.2,
# }

# # —— 仅对某一类把文字左移若干“字符宽” —— 
# SHIFT_CLASS = 'vegetation'  # 只移动植被类
# SHIFT_CHARS = 2             # 向左移动 2 个字符宽

# # 1) 模型（按需改成你的权重路径）
# model = YOLO(r'/home/xgq/Desktop/yolo/runs/train/exp6/weights/best.pt')

# # 2) 颜色（deadmood=洋红，更醒目）
# CUSTOM_COLORS_RGB = {
#     'brick_loss': (0, 255, 255),   # 青色
#     'vegetation': (0, 0, 255),     # 蓝色
#     'deadmood':  (255, 0, 255),    # 洋红
# }
# CUSTOM_COLORS_BGR = {k: (v[2], v[1], v[0]) for k, v in CUSTOM_COLORS_RGB.items()}
# _ultra_colors = Colors()

# def pick_color_bgr(cls_id: int, name: str):
#     name_l = name.lower() if isinstance(name, str) else str(name).lower()
#     if name_l in CUSTOM_COLORS_BGR:
#         return CUSTOM_COLORS_BGR[name_l]
#     c = _ultra_colors(cls_id, bgr=True)
#     return (int(c[0]), int(c[1]), int(c[2]))

# def draw_ultra_style_custom(im_bgr: np.ndarray, boxes_xyxy: np.ndarray,
#                             scores: np.ndarray, labels: np.ndarray,
#                             names_map: dict) -> np.ndarray:
#     """
#     不在此处做阈值判断；传进来的就是需要绘制的框。
#     显示分数 = 真实分数（不做加值/封顶）。
#     'vegetation' 的文字整体向左移动 SHIFT_CHARS 个字符宽。
#     """
#     H, W = im_bgr.shape[:2]
#     lw = max(round((H + W) / 2 * 0.003), 2)
#     tf = max(lw - 1, 1)
#     fs = lw / 3
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     # 用“0”估算字符宽度，然后乘 SHIFT_CHARS
#     ref_text = "0" * max(1, int(SHIFT_CHARS))
#     (shift_width, _), _ = cv2.getTextSize(ref_text, font, fs, tf)

#     for i in range(len(boxes_xyxy)):
#         x1, y1, x2, y2 = [int(round(v)) for v in boxes_xyxy[i].tolist()]
#         show_conf = float(scores[i])  # 真实分数

#         cls_id = int(labels[i])
#         name   = names_map.get(cls_id, str(cls_id))
#         name_l = name.lower() if isinstance(name, str) else str(name).lower()

#         color = pick_color_bgr(cls_id, name_l)
#         txt_color = (0, 0, 0) if name_l in {'brick_loss', 'deadmood'} else (255, 255, 255)

#         # 画框
#         cv2.rectangle(im_bgr, (x1, y1), (x2, y2), color, thickness=lw)

#         # 文本与背景条
#         label = f'{name} {show_conf:.2f}'
#         (tw, th), base = cv2.getTextSize(label, font, fs, tf)
#         y_text = max(y1, th + base + 3)

#         x_text = x1 + 1
#         if name_l == SHIFT_CLASS:
#             x_text = max(x_text - shift_width, 0)

#         bg_left  = max(x_text - 1, 0)
#         bg_right = min(x_text + tw + 1, W - 1)
#         cv2.rectangle(im_bgr, (bg_left, y_text - th - base), (bg_right, y_text), color, -1)

#         cv2.putText(im_bgr, label, (x_text, y_text - base),
#                     font, fs, txt_color, thickness=tf, lineType=cv2.LINE_AA)
#     return im_bgr

# # 3) 简单 IoU 和 NMS（xyxy）
# def iou_xyxy(box, boxes):
#     x1 = np.maximum(box[0], boxes[:, 0])
#     y1 = np.maximum(box[1], boxes[:, 1])
#     x2 = np.minimum(box[2], boxes[:, 2])
#     y2 = np.minimum(box[3], boxes[:, 3])
#     inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
#     area1 = (box[2] - box[0]) * (box[3] - box[1])
#     area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#     union = area1 + area2 - inter + 1e-9
#     return inter / union

# def nms_xyxy(xyxy, scores, iou_thr=0.5):
#     if len(xyxy) == 0:
#         return np.array([], dtype=int)
#     order = scores.argsort()[::-1]
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         if order.size == 1:
#             break
#         ious = iou_xyxy(xyxy[i], xyxy[order[1:]])
#         order = order[1:][ious < iou_thr]
#     return np.array(keep, dtype=int)

# def classwise_filter(xyxy, scores, labels, names, class_conf, class_iou):
#     keep_all = []
#     unique_c = np.unique(labels)
#     for cid in unique_c:
#         name = names.get(int(cid), str(int(cid))).lower()
#         conf_thr = class_conf.get(name, 0.0)
#         iou_thr  = class_iou.get(name, 0.7)
#         m = (labels == cid) & (scores >= conf_thr)
#         if not m.any():
#             continue
#         k = nms_xyxy(xyxy[m], scores[m], iou_thr=iou_thr)
#         keep_idx = np.where(m)[0][k]
#         keep_all.extend(keep_idx.tolist())
#     keep_all = np.array(keep_all, dtype=int)
#     return xyxy[keep_all], scores[keep_all], labels[keep_all]

# # 4) 输入“文件夹”路径
# base_path = r"/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/dataset/6.2/images/test"
# assert os.path.isdir(base_path), f"找不到文件夹：{base_path}"

# # 收集该文件夹下的所有常见图片
# exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
# image_paths = sorted([str(p) for p in Path(base_path).iterdir() if p.suffix.lower() in exts])
# assert image_paths, f"在 {base_path} 下未找到任何图片文件"

# # 5) 输出目录（只保留一份结果）
# save_base_dir = 'runs/detect/CWADE-Net'
# os.makedirs(save_base_dir, exist_ok=True)

# # 6) 预测（放宽阈值，避免提前被过滤/NMS）
# results = model.predict(source=image_paths, conf=BASE_CONF, iou=BASE_IOU, save=False)

# for result in results:
#     path = getattr(result, 'orig_img_path', None) or getattr(result, 'path', None)
#     img_name = os.path.basename(path) if path else 'unknown.jpg'

#     # names 统一成 dict
#     names_attr = getattr(model, 'names', {})
#     names = names_attr if isinstance(names_attr, dict) else {i: n for i, n in enumerate(list(names_attr))}

#     boxes = result.boxes
#     print(f"\n—— 图片：{img_name} ——")
#     if boxes is None or len(boxes) == 0:
#         print("未检测到任何目标。")
#         im0 = result.orig_img if hasattr(result, 'orig_img') else cv2.imread(path)
#         if im0 is not None:
#             cv2.imwrite(os.path.join(save_base_dir, img_name), im0)
#         continue

#     # 原始预测（放宽阈值后）
#     cls_all  = boxes.cls.int().cpu().numpy()
#     conf_all = boxes.conf.cpu().numpy()
#     xyxy_all = boxes.xyxy.cpu().numpy()
#     xywh_all = boxes.xywh.cpu().numpy()

#     for i in range(len(xyxy_all)):
#         cname = names.get(int(cls_all[i]), str(int(cls_all[i])))
#         print(f"[RAW] 类别:{cname} | 置信度:{float(conf_all[i]):.3f} | xywh:{xywh_all[i].tolist()}")

#     # 按类别做 conf 过滤 + 按类别 NMS
#     xyxy_keep, conf_keep, cls_keep = classwise_filter(
#         xyxy_all, conf_all, cls_all, names, CLASS_CONF, CLASS_IOU
#     )

#     # 打印过滤后的结果（真实分数）
#     for i in range(len(xyxy_keep)):
#         cname = names.get(int(cls_keep[i]), str(int(cls_keep[i])))
#         print(f"[KEPT] 类别:{cname} | 置信度:{float(conf_keep[i]):.3f} | xyxy:{xyxy_keep[i].tolist()}")

#     # 绘制并保存（显示真实分数；vegetation 的文字左移）
#     im0 = result.orig_img if hasattr(result, 'orig_img') else cv2.imread(path)
#     if im0 is None:
#         continue
#     im_drawn = draw_ultra_style_custom(im0.copy(), xyxy_keep, conf_keep, cls_keep, names)
#     cv2.imwrite(os.path.join(save_base_dir, img_name), im_drawn)

# print("\n✅ 保存完成：", os.path.abspath(save_base_dir))



# 5 对4的升级版，要求输入既可以是图片也可以是文件夹
# from ultralytics import YOLO
# from ultralytics.utils.plotting import Colors
# from pathlib import Path
# import os, cv2, numpy as np
# from glob import glob

# # ==============================
# # 0) 全局“放宽”阈值（先尽量多拿框）
# # ==============================
# BASE_CONF = 0.001   # 放宽，几乎不过滤
# BASE_IOU  = 0.99    # 放宽，尽量不抑制

# # —— 每类各自的筛选阈值（影响保留框 & NMS）——
# CLASS_CONF = {        # 各类别最小置信度（真实分数）
#     'brick_loss': 0.2,
#     'vegetation': 0.2,
#     'deadmood':  0.3,
# }
# CLASS_IOU = {         # 各类别 NMS 的 IoU 阈值
#     'brick_loss': 0.3,
#     'vegetation': 0.2,
#     'deadmood':  0.3,
# }

# # —— 仅对某一类把文字左移若干“字符宽” —— 
# SHIFT_CLASS = 'vegetation'  # 只移动植被类
# SHIFT_CHARS = 2             # 向左移动 2 个字符宽

# # 1) 模型（按需改成你的权重路径）
# model = YOLO(r'/home/xgq/Desktop/yolo/runs/train/exp3/weights/best.pt')

# # 2) 颜色（deadmood=洋红，更醒目）
# CUSTOM_COLORS_RGB = {
#     'brick_loss': (0, 255, 255),   # 青色
#     'vegetation': (0, 0, 255),     # 蓝色
#     'deadmood':  (255, 0, 255),    # 洋红
# }
# CUSTOM_COLORS_BGR = {k: (v[2], v[1], v[0]) for k, v in CUSTOM_COLORS_RGB.items()}
# _ultra_colors = Colors()

# def pick_color_bgr(cls_id: int, name: str):
#     name_l = name.lower() if isinstance(name, str) else str(name).lower()
#     if name_l in CUSTOM_COLORS_BGR:
#         return CUSTOM_COLORS_BGR[name_l]
#     c = _ultra_colors(cls_id, bgr=True)
#     return (int(c[0]), int(c[1]), int(c[2]))

# def draw_ultra_style_custom(im_bgr: np.ndarray, boxes_xyxy: np.ndarray,
#                             scores: np.ndarray, labels: np.ndarray,
#                             names_map: dict) -> np.ndarray:
#     """
#     不在此处做阈值判断；传进来的就是需要绘制的框。
#     显示分数 = 真实分数（不做加值/封顶）。
#     'vegetation' 的文字整体向左移动 SHIFT_CHARS 个字符宽。
#     """
#     H, W = im_bgr.shape[:2]
#     lw = max(round((H + W) / 2 * 0.003), 2)
#     tf = max(lw - 1, 1)
#     fs = lw / 3
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     # 用“0”估算字符宽度，然后乘 SHIFT_CHARS
#     ref_text = "0" * max(1, int(SHIFT_CHARS))
#     (shift_width, _), _ = cv2.getTextSize(ref_text, font, fs, tf)

#     for i in range(len(boxes_xyxy)):
#         x1, y1, x2, y2 = [int(round(v)) for v in boxes_xyxy[i].tolist()]
#         show_conf = float(scores[i])  # 真实分数

#         cls_id = int(labels[i])
#         name   = names_map.get(cls_id, str(cls_id))
#         name_l = name.lower() if isinstance(name, str) else str(name).lower()

#         color = pick_color_bgr(cls_id, name_l)
#         txt_color = (0, 0, 0) if name_l in {'brick_loss', 'deadmood'} else (255, 255, 255)

#         # 画框
#         cv2.rectangle(im_bgr, (x1, y1), (x2, y2), color, thickness=lw)

#         # 文本与背景条
#         label = f'{name} {show_conf:.2f}'
#         (tw, th), base = cv2.getTextSize(label, font, fs, tf)
#         y_text = max(y1, th + base + 3)

#         x_text = x1 + 1
#         if name_l == SHIFT_CLASS:
#             x_text = max(x_text - shift_width, 0)

#         bg_left  = max(x_text - 1, 0)
#         bg_right = min(x_text + tw + 1, W - 1)
#         cv2.rectangle(im_bgr, (bg_left, y_text - th - base), (bg_right, y_text), color, -1)

#         cv2.putText(im_bgr, label, (x_text, y_text - base),
#                     font, fs, txt_color, thickness=tf, lineType=cv2.LINE_AA)
#     return im_bgr

# # 3) 简单 IoU 和 NMS（xyxy）
# def iou_xyxy(box, boxes):
#     x1 = np.maximum(box[0], boxes[:, 0])
#     y1 = np.maximum(box[1], boxes[:, 1])
#     x2 = np.minimum(box[2], boxes[:, 2])
#     y2 = np.minimum(box[3], boxes[:, 3])
#     inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
#     area1 = (box[2] - box[0]) * (box[3] - box[1])
#     area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#     union = area1 + area2 - inter + 1e-9
#     return inter / union

# def nms_xyxy(xyxy, scores, iou_thr=0.5):
#     if len(xyxy) == 0:
#         return np.array([], dtype=int)
#     order = scores.argsort()[::-1]
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         if order.size == 1:
#             break
#         ious = iou_xyxy(xyxy[i], xyxy[order[1:]])
#         order = order[1:][ious < iou_thr]
#     return np.array(keep, dtype=int)

# def classwise_filter(xyxy, scores, labels, names, class_conf, class_iou):
#     keep_all = []
#     unique_c = np.unique(labels)
#     for cid in unique_c:
#         name = names.get(int(cid), str(int(cid))).lower()
#         conf_thr = class_conf.get(name, 0.0)
#         iou_thr  = class_iou.get(name, 0.7)
#         m = (labels == cid) & (scores >= conf_thr)
#         if not m.any():
#             continue
#         k = nms_xyxy(xyxy[m], scores[m], iou_thr=iou_thr)
#         keep_idx = np.where(m)[0][k]
#         keep_all.extend(keep_idx.tolist())
#     keep_all = np.array(keep_all, dtype=int)
#     return xyxy[keep_all], scores[keep_all], labels[keep_all]

# # 4) 输入路径：既支持“文件夹”，也支持“单图”或“通配符”
# #    例（文件夹）：/home/xgq/.../images/test
# #    例（单图）：  /home/xgq/.../images/test/abc.jpg
# #    例（通配）：  /home/xgq/.../images/test/*.jpg
# # input_path = r"/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/dataset/6.2/images/val/(1659).JPG"

# input_path = r"/home/xgq/Desktop/DJI/DJI_202511020824_004/DJI_20251102082631_0001_V.JPG"

# def gather_images(p):
#     exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
#     if os.path.isdir(p):
#         paths = sorted(str(x) for x in Path(p).iterdir() if x.suffix.lower() in exts)
#         return paths
#     if os.path.isfile(p):
#         assert Path(p).suffix.lower() in exts, f"不支持的图片后缀：{p}"
#         return [p]
#     # 通配符
#     paths = sorted(q for q in glob(p) if Path(q).suffix.lower() in exts)
#     return paths

# image_paths = gather_images(input_path)
# assert image_paths, f"未找到图片：{input_path}"

# # 5) 输出目录（同一目录下自动避免覆盖）
# save_base_dir = 'runs/detect/CWADE-Net'
# os.makedirs(save_base_dir, exist_ok=True)

# def save_with_suffix_if_exists(save_dir, img_name, img):
#     """
#     如果同名文件已存在，则自动在文件名后加 _1, _2, ... 避免覆盖
#     """
#     stem, ext = os.path.splitext(img_name)
#     out_path = os.path.join(save_dir, img_name)
#     idx = 1
#     while os.path.exists(out_path):
#         out_path = os.path.join(save_dir, f"{stem}_{idx}{ext}")
#         idx += 1
#     cv2.imwrite(out_path, img)
#     print("保存到：", out_path)

# # 6) 预测（放宽阈值，避免提前被过滤/NMS）
# results = model.predict(source=image_paths, conf=BASE_CONF, iou=BASE_IOU, save=False)

# for result in results:
#     path = getattr(result, 'orig_img_path', None) or getattr(result, 'path', None)
#     img_name = os.path.basename(path) if path else 'unknown.jpg'

#     # names 统一成 dict
#     names_attr = getattr(model, 'names', {})
#     names = names_attr if isinstance(names_attr, dict) else {i: n for i, n in enumerate(list(names_attr))}

#     boxes = result.boxes
#     print(f"\n—— 图片：{img_name} ——")
#     if boxes is None or len(boxes) == 0:
#         print("未检测到任何目标。")
#         im0 = result.orig_img if hasattr(result, 'orig_img') else cv2.imread(path)
#         if im0 is not None:
#             save_with_suffix_if_exists(save_base_dir, img_name, im0)
#         continue

#     # 原始预测（放宽阈值后）
#     cls_all  = boxes.cls.int().cpu().numpy()
#     conf_all = boxes.conf.cpu().numpy()
#     xyxy_all = boxes.xyxy.cpu().numpy()
#     xywh_all = boxes.xywh.cpu().numpy()

#     for i in range(len(xyxy_all)):
#         cname = names.get(int(cls_all[i]), str(int(cls_all[i])))
#         print(f"[RAW] 类别:{cname} | 置信度:{float(conf_all[i]):.3f} | xywh:{xywh_all[i].tolist()}")

#     # 按类别做 conf 过滤 + 按类别 NMS
#     xyxy_keep, conf_keep, cls_keep = classwise_filter(
#         xyxy_all, conf_all, cls_all, names, CLASS_CONF, CLASS_IOU
#     )

#     # 打印过滤后的结果（真实分数）
#     for i in range(len(xyxy_keep)):
#         cname = names.get(int(cls_keep[i]), str(int(cls_keep[i])))
#         print(f"[KEPT] 类别:{cname} | 置信度:{float(conf_keep[i]):.3f} | xyxy:{xyxy_keep[i].tolist()}")

#     # 绘制并保存（显示真实分数；vegetation 的文字左移）
#     im0 = result.orig_img if hasattr(result, 'orig_img') else cv2.imread(path)
#     if im0 is None:
#         continue
#     im_drawn = draw_ultra_style_custom(im0.copy(), xyxy_keep, conf_keep, cls_keep, names)
#     save_with_suffix_if_exists(save_base_dir, img_name, im_drawn)

# print("\n✅ 保存完成：", os.path.abspath(save_base_dir))
# print(f"共处理图片数：{len(image_paths)}")








#6,在5的基础上增加类别名称显示开关
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors
from pathlib import Path
import os, cv2, numpy as np
from glob import glob

# ==============================
# 0) 全局“放宽”阈值（先尽量多拿框）
# ==============================
BASE_CONF = 0.001   # 放宽，几乎不过滤
BASE_IOU  = 0.99    # 放宽，尽量不抑制

# —— 每类各自的筛选阈值（影响保留框 & NMS）——
CLASS_CONF = {        # 各类别最小置信度（真实分数）
    'brick_loss': 0.1,
    'vegetation': 0.25,
    'deadmood':  0.1,
}
CLASS_IOU = {         # 各类别 NMS 的 IoU 阈值
    'brick_loss': 0.5,
    'vegetation': 0.1,
    'deadmood':  0.1,
}

# —— 仅对某一类把文字左移若干“字符宽” —— 
SHIFT_CLASS = 'vegetation'  # 只移动植被类
SHIFT_CHARS = 2             # 向左移动 2 个字符宽

# ==============================
# 文字显示配置（新增）
# ==============================
# 全局默认显示模式：'name_score'（显示“类名+分数”）| 'score_only'（只显示分数）| 'none'（不显示文字）
LABEL_GLOBAL_MODE = 'name_score'

# 可按类覆盖显示模式；不在此字典内的类使用 LABEL_GLOBAL_MODE
LABEL_CLASS_MODE = {
    # 'brick_loss': 'score_only',
    # 'vegetation': 'none',
    # 'deadmood': 'name_score',
}

SCORE_DECIMALS = 2  # 分数保留小数位数

# 1) 模型（按需改成你的权重路径）
model = YOLO(r'/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/runs/train/exp394/weights/best.pt')
# model = YOLO(r'/mnt/sda2/yolo/ultralytics-8.3.39/runs/train/exp8/weights/best.pt')

# 2) 颜色（deadmood=洋红，更醒目）
CUSTOM_COLORS_RGB = {
    'brick_loss': (0, 255, 255),   # 青色
    'vegetation': (0, 0, 255),     # 蓝色
    'deadmood':  (255, 0, 255),    # 洋红
}
CUSTOM_COLORS_BGR = {k: (v[2], v[1], v[0]) for k, v in CUSTOM_COLORS_RGB.items()}
_ultra_colors = Colors()

def pick_color_bgr(cls_id: int, name: str):
    name_l = name.lower() if isinstance(name, str) else str(name).lower()
    if name_l in CUSTOM_COLORS_BGR:
        return CUSTOM_COLORS_BGR[name_l]
    c = _ultra_colors(cls_id, bgr=True)
    return (int(c[0]), int(c[1]), int(c[2]))

def get_label_mode_for_class(name_l: str) -> str:
    """返回该类应使用的显示模式：'name_score' | 'score_only' | 'none'"""
    return LABEL_CLASS_MODE.get(name_l, LABEL_GLOBAL_MODE)

def build_label_text(name_l: str, score: float) -> str:
    mode = get_label_mode_for_class(name_l)
    if mode == 'none':
        return ''
    s = f'{score:.{SCORE_DECIMALS}f}'
    if mode == 'score_only':
        return s
    # name_score
    return f'{name_l} {s}'

def draw_ultra_style_custom(im_bgr: np.ndarray, boxes_xyxy: np.ndarray,
                            scores: np.ndarray, labels: np.ndarray,
                            names_map: dict) -> np.ndarray:
    """
    不在此处做阈值判断；传进来的就是需要绘制的框。
    显示分数 = 真实分数（不做加值/封顶）。
    文字显示受 LABEL_GLOBAL_MODE / LABEL_CLASS_MODE 控制。
    'vegetation' 的文字仅在显示文字时向左移动 SHIFT_CHARS 个字符宽。
    """
    H, W = im_bgr.shape[:2]
    lw = max(round((H + W) / 2 * 0.003), 2)
    tf = max(lw - 1, 1)
    fs = lw / 3
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 用“0”估算字符宽度，然后乘 SHIFT_CHARS
    ref_text = "0" * max(1, int(SHIFT_CHARS))
    (shift_width, _), _ = cv2.getTextSize(ref_text, font, fs, tf)

    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = [int(round(v)) for v in boxes_xyxy[i].tolist()]
        show_conf = float(scores[i])  # 真实分数

        cls_id = int(labels[i])
        name   = names_map.get(cls_id, str(cls_id))
        name_l = name.lower() if isinstance(name, str) else str(name).lower()

        color = pick_color_bgr(cls_id, name_l)
        # 针对不同底色选文字色
        txt_color = (0, 0, 0) if name_l in {'brick_loss', 'deadmood'} else (255, 255, 255)

        # 画框
        cv2.rectangle(im_bgr, (x1, y1), (x2, y2), color, thickness=lw)

        # 构建文本
        label = build_label_text(name_l, show_conf)
        if not label:   # 'none' 模式：不绘制文字与背景
            continue

        (tw, th), base = cv2.getTextSize(label, font, fs, tf)
        y_text = max(y1, th + base + 3)

        x_text = x1 + 1
        # 只有在显示文字时才考虑左移
        if name_l == SHIFT_CLASS:
            x_text = max(x_text - shift_width, 0)

        bg_left  = max(x_text - 1, 0)
        bg_right = min(x_text + tw + 1, W - 1)
        cv2.rectangle(im_bgr, (bg_left, y_text - th - base), (bg_right, y_text), color, -1)
        cv2.putText(im_bgr, label, (x_text, y_text - base),
                    font, fs, txt_color, thickness=tf, lineType=cv2.LINE_AA)
    return im_bgr

# 3) 简单 IoU 和 NMS（xyxy）
def iou_xyxy(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-9
    return inter / union

def nms_xyxy(xyxy, scores, iou_thr=0.5):
    if len(xyxy) == 0:
        return np.array([], dtype=int)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_xyxy(xyxy[i], xyxy[order[1:]])
        order = order[1:][ious < iou_thr]
    return np.array(keep, dtype=int)

def classwise_filter(xyxy, scores, labels, names, class_conf, class_iou):
    keep_all = []
    unique_c = np.unique(labels)
    for cid in unique_c:
        name = names.get(int(cid), str(int(cid))).lower()
        conf_thr = class_conf.get(name, 0.0)
        iou_thr  = class_iou.get(name, 0.7)
        m = (labels == cid) & (scores >= conf_thr)
        if not m.any():
            continue
        k = nms_xyxy(xyxy[m], scores[m], iou_thr=iou_thr)
        keep_idx = np.where(m)[0][k]
        keep_all.extend(keep_idx.tolist())
    keep_all = np.array(keep_all, dtype=int)
    return xyxy[keep_all], scores[keep_all], labels[keep_all]

# 4) 输入路径：既支持“文件夹”，也支持“单图”或“通配符”
input_path = r"/home/xgq/Desktop/天气光照实验/运动模糊/55模糊/1.png"
# input_path = r'/home/xgq/Desktop/天气光照实验/运动模糊/55模糊'

def gather_images(p):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    if os.path.isdir(p):
        paths = sorted(str(x) for x in Path(p).iterdir() if x.suffix.lower() in exts)
        return paths
    if os.path.isfile(p):
        assert Path(p).suffix.lower() in exts, f"不支持的图片后缀：{p}"
        return [p]
    # 通配符
    paths = sorted(q for q in glob(p) if Path(q).suffix.lower() in exts)
    return paths

image_paths = gather_images(input_path)
assert image_paths, f"未找到图片：{input_path}"

# 5) 输出目录（只保留一份结果）
save_base_dir = 'runs/detect/45'
os.makedirs(save_base_dir, exist_ok=True)

# 6) 预测（放宽阈值，避免提前被过滤/NMS）
results = model.predict(source=input_path, conf=BASE_CONF, iou=BASE_IOU, save=False)

for result in results:
    path = getattr(result, 'orig_img_path', None) or getattr(result, 'path', None)
    img_name = os.path.basename(path) if path else 'unknown.jpg'

    # names 统一成 dict
    names_attr = getattr(model, 'names', {})
    names = names_attr if isinstance(names_attr, dict) else {i: n for i, n in enumerate(list(names_attr))}

    boxes = result.boxes
    print(f"\n—— 图片：{img_name} ——")
    if boxes is None or len(boxes) == 0:
        print("未检测到任何目标。")
        im0 = result.orig_img if hasattr(result, 'orig_img') else cv2.imread(path)
        if im0 is not None:
            cv2.imwrite(os.path.join(save_base_dir, img_name), im0)
        continue

    # 原始预测（放宽阈值后）
    cls_all  = boxes.cls.int().cpu().numpy()
    conf_all = boxes.conf.cpu().numpy()
    xyxy_all = boxes.xyxy.cpu().numpy()
    xywh_all = boxes.xywh.cpu().numpy()

    for i in range(len(xyxy_all)):
        cname = names.get(int(cls_all[i]), str(int(cls_all[i])))
        print(f"[RAW] 类别:{cname} | 置信度:{float(conf_all[i]):.{SCORE_DECIMALS}f} | xywh:{xywh_all[i].tolist()}")

    # 按类别做 conf 过滤 + 按类别 NMS
    xyxy_keep, conf_keep, cls_keep = classwise_filter(
        xyxy_all, conf_all, cls_all, names, CLASS_CONF, CLASS_IOU
    )

    # 打印过滤后的结果（真实分数）
    for i in range(len(xyxy_keep)):
        cname = names.get(int(cls_keep[i]), str(int(cls_keep[i])))
        print(f"[KEPT] 类别:{cname} | 置信度:{float(conf_keep[i]):.{SCORE_DECIMALS}f} | xyxy:{xyxy_keep[i].tolist()}")

    # 绘制并保存（显示真实分数；vegetation 的文字左移）
    im0 = result.orig_img if hasattr(result, 'orig_img') else cv2.imread(path)
    if im0 is None:
        continue
    im_drawn = draw_ultra_style_custom(im0.copy(), xyxy_keep, conf_keep, cls_keep, names)
    cv2.imwrite(os.path.join(save_base_dir, img_name), im_drawn)

print("\n✅ 保存完成：", os.path.abspath(save_base_dir))
print(f"共处理图片数：{len(image_paths)}")

# from ultralytics import YOLO
# from ultralytics.utils.plotting import Colors
# from pathlib import Path
# import os
# import cv2
# import numpy as np
# from glob import glob

# # ==============================
# # 0) 全局“放宽”阈值（先尽量多拿框）
# # ==============================
# BASE_CONF = 0.1   # 预测时尽量不过滤
# BASE_IOU  = 0.5    # 预测时尽量不做强 NMS

# # ==============================
# # 1) 单类别筛选阈值（你的类别只有 crack）
# # ==============================
# CRACK_CONF = 0.10   # crack 类最小保留置信度
# CRACK_IOU  = 0.10   # crack 类 NMS 的 IoU 阈值

# # ==============================
# # 2) 文字显示配置
# # 'name_score'：显示“crack 0.95”
# # 'score_only'：只显示“0.95”
# # 'none'      ：不显示文字
# # ==============================
# LABEL_MODE = 'name_score'
# SCORE_DECIMALS = 2  # 分数保留小数位数

# # ==============================
# # 3) 模型路径（改成你的权重）
# # ==============================
# model = YOLO(r'/home/xgq/Desktop/yolo/runs/train/exp93/weights/best.pt')

# # ==============================
# # 4) crack 类颜色设置
# # RGB: (255, 0, 255) = 洋红
# # OpenCV 画图用 BGR，所以转换一下
# # ==============================
# CRACK_COLOR_RGB = (0, 255, 0)
# CRACK_COLOR_BGR = (CRACK_COLOR_RGB[2], CRACK_COLOR_RGB[1], CRACK_COLOR_RGB[0])

# _ultra_colors = Colors()

# def pick_color_bgr(cls_id: int, name: str):
#     """单类别优先固定用 crack 的颜色，否则退回 ultralytics 默认颜色。"""
#     name_l = str(name).lower()
#     if name_l == 'crack':
#         return CRACK_COLOR_BGR
#     c = _ultra_colors(cls_id, bgr=True)
#     return (int(c[0]), int(c[1]), int(c[2]))

# def build_label_text(name_l: str, score: float) -> str:
#     """构建标签文字。"""
#     if LABEL_MODE == 'none':
#         return ''
#     s = f'{score:.{SCORE_DECIMALS}f}'
#     if LABEL_MODE == 'score_only':
#         return s
#     return f'{name_l} {s}'

# def draw_ultra_style_custom(im_bgr: np.ndarray,
#                             boxes_xyxy: np.ndarray,
#                             scores: np.ndarray,
#                             labels: np.ndarray,
#                             names_map: dict) -> np.ndarray:
#     """
#     绘制检测框和文字。
#     传进来的 boxes/scores/labels 已经是筛选后的结果。
#     """
#     H, W = im_bgr.shape[:2]
#     lw = max(round((H + W) / 2 * 0.003), 2)   # 线宽
#     tf = max(lw - 1, 1)                       # 字体粗细
#     fs = lw / 3                               # 字体大小
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     for i in range(len(boxes_xyxy)):
#         x1, y1, x2, y2 = [int(round(v)) for v in boxes_xyxy[i].tolist()]
#         score = float(scores[i])

#         cls_id = int(labels[i])
#         name = names_map.get(cls_id, str(cls_id))
#         name_l = str(name).lower()

#         color = pick_color_bgr(cls_id, name_l)
#         txt_color = (255, 255, 255)  # 白字

#         # 画框
#         cv2.rectangle(im_bgr, (x1, y1), (x2, y2), color, thickness=lw)

#         # 构建文字
#         label = build_label_text(name_l, score)
#         if not label:
#             continue

#         (tw, th), base = cv2.getTextSize(label, font, fs, tf)
#         y_text = max(y1, th + base + 3)
#         x_text = max(x1 + 1, 0)

#         # 文字背景框
#         bg_left = max(x_text - 1, 0)
#         bg_right = min(x_text + tw + 1, W - 1)
#         bg_top = max(y_text - th - base, 0)
#         bg_bottom = min(y_text, H - 1)

#         cv2.rectangle(im_bgr, (bg_left, bg_top), (bg_right, bg_bottom), color, -1)
#         cv2.putText(im_bgr, label, (x_text, y_text - base),
#                     font, fs, txt_color, thickness=tf, lineType=cv2.LINE_AA)

#     return im_bgr

# # ==============================
# # 5) IoU 和 NMS
# # ==============================
# def iou_xyxy(box, boxes):
#     x1 = np.maximum(box[0], boxes[:, 0])
#     y1 = np.maximum(box[1], boxes[:, 1])
#     x2 = np.minimum(box[2], boxes[:, 2])
#     y2 = np.minimum(box[3], boxes[:, 3])

#     inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
#     area1 = (box[2] - box[0]) * (box[3] - box[1])
#     area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#     union = area1 + area2 - inter + 1e-9
#     return inter / union

# def nms_xyxy(xyxy, scores, iou_thr=0.5):
#     if len(xyxy) == 0:
#         return np.array([], dtype=int)

#     order = scores.argsort()[::-1]
#     keep = []

#     while order.size > 0:
#         i = order[0]
#         keep.append(i)

#         if order.size == 1:
#             break

#         ious = iou_xyxy(xyxy[i], xyxy[order[1:]])
#         order = order[1:][ious < iou_thr]

#     return np.array(keep, dtype=int)

# def crack_filter(xyxy, scores, labels, names_map):
#     """
#     单类别筛选：
#     1. 只保留 crack 类
#     2. conf >= CRACK_CONF
#     3. 对 crack 做 NMS
#     """
#     keep_mask = []

#     for i in range(len(labels)):
#         cls_id = int(labels[i])
#         name = str(names_map.get(cls_id, cls_id)).lower()
#         keep_mask.append(name == 'crack' and scores[i] >= CRACK_CONF)

#     keep_mask = np.array(keep_mask, dtype=bool)

#     if not keep_mask.any():
#         return (
#             np.empty((0, 4), dtype=np.float32),
#             np.empty((0,), dtype=np.float32),
#             np.empty((0,), dtype=np.int32),
#         )

#     xyxy_keep = xyxy[keep_mask]
#     scores_keep = scores[keep_mask]
#     labels_keep = labels[keep_mask]

#     nms_idx = nms_xyxy(xyxy_keep, scores_keep, iou_thr=CRACK_IOU)
#     return xyxy_keep[nms_idx], scores_keep[nms_idx], labels_keep[nms_idx]

# # ==============================
# # 6) 输入路径：支持单图 / 文件夹 / 通配符
# # ==============================
# # input_path = r"/home/xgq/Desktop/裂缝数据集/1/496.png"
# input_path = r"/home/xgq/Desktop/裂缝数据集/images/test"

# def gather_images(p):
#     exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
#     if os.path.isdir(p):
#         return sorted(str(x) for x in Path(p).iterdir() if x.suffix.lower() in exts)
#     if os.path.isfile(p):
#         assert Path(p).suffix.lower() in exts, f"不支持的图片后缀：{p}"
#         return [p]
#     return sorted(q for q in glob(p) if Path(q).suffix.lower() in exts)

# image_paths = gather_images(input_path)
# assert image_paths, f"未找到图片：{input_path}"

# # ==============================
# # 7) 输出目录
# # ==============================
# # save_base_dir = 'runs/detect/test93'
# save_base_dir = '/home/xgq/Desktop/裂缝数据集/检测结果'
# os.makedirs(save_base_dir, exist_ok=True)

# # ==============================
# # 8) 预测
# # ==============================
# results = model.predict(
#     source=input_path,
#     conf=BASE_CONF,
#     iou=BASE_IOU,
#     save=False,
#     batch=1,
#     imgsz=512,
#     stream=True,
#     device=0
# )

# for result in results:
#     path = getattr(result, 'orig_img_path', None) or getattr(result, 'path', None)
#     img_name = os.path.basename(path) if path else 'unknown.jpg'

#     # names 统一成 dict
#     names_attr = getattr(model, 'names', {})
#     names = names_attr if isinstance(names_attr, dict) else {i: n for i, n in enumerate(list(names_attr))}

#     boxes = result.boxes
#     print(f"\n—— 图片：{img_name} ——")

#     if boxes is None or len(boxes) == 0:
#         print("未检测到任何目标。")
#         im0 = result.orig_img if hasattr(result, 'orig_img') else cv2.imread(path)
#         if im0 is not None:
#             cv2.imwrite(os.path.join(save_base_dir, img_name), im0)
#         continue

#     # 原始预测结果
#     cls_all = boxes.cls.int().cpu().numpy()
#     conf_all = boxes.conf.cpu().numpy()
#     xyxy_all = boxes.xyxy.cpu().numpy()
#     xywh_all = boxes.xywh.cpu().numpy()

#     for i in range(len(xyxy_all)):
#         cname = names.get(int(cls_all[i]), str(int(cls_all[i])))
#         print(f"[RAW] 类别:{cname} | 置信度:{float(conf_all[i]):.{SCORE_DECIMALS}f} | xywh:{xywh_all[i].tolist()}")

#     # 单类别 crack 筛选
#     xyxy_keep, conf_keep, cls_keep = crack_filter(
#         xyxy_all, conf_all, cls_all, names
#     )

#     if len(xyxy_keep) == 0:
#         print("[KEPT] 过滤后无保留目标。")
#     else:
#         for i in range(len(xyxy_keep)):
#             cname = names.get(int(cls_keep[i]), str(int(cls_keep[i])))
#             print(f"[KEPT] 类别:{cname} | 置信度:{float(conf_keep[i]):.{SCORE_DECIMALS}f} | xyxy:{xyxy_keep[i].tolist()}")

#     # 绘制并保存
#     im0 = result.orig_img if hasattr(result, 'orig_img') else cv2.imread(path)
#     if im0 is None:
#         continue

#     im_drawn = draw_ultra_style_custom(im0.copy(), xyxy_keep, conf_keep, cls_keep, names)
#     cv2.imwrite(os.path.join(save_base_dir, img_name), im_drawn)

# print("\n✅ 保存完成：", os.path.abspath(save_base_dir))
# print(f"共处理图片数：{len(image_paths)}")