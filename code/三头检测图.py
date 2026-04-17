import os
import cv2
import torch
from ultralytics import YOLO

# ==============================
# 文字 & 颜色风格配置（与另一份脚本保持一致）
# ==============================

# 每类颜色：先按 RGB 定义，再转 BGR
CUSTOM_COLORS_RGB = {
    'brick_loss': (0, 255, 255),   # 青色
    'vegetation': (0, 0, 255),     # 蓝色
    'deadmood':  (255, 0, 255),    # 洋红/紫
}
CUSTOM_COLORS_BGR = {k: (v[2], v[1], v[0]) for k, v in CUSTOM_COLORS_RGB.items()}

# 只对某一类左移文字
SHIFT_CLASS = 'vegetation'  # 只移动植被类
SHIFT_CHARS = 2             # 向左移动 2 个字符宽

# 文字显示模式
# 'name_score'：显示“类名+分数”
# 'score_only'：只显示分数
# 'none'      ：不显示文字
LABEL_GLOBAL_MODE = 'name_score'
LABEL_CLASS_MODE = {
    # 可以按类单独改模式，例如：
    # 'brick_loss': 'score_only',
    # 'vegetation': 'none',
}
SCORE_DECIMALS = 2  # 分数保留小数位

def get_label_mode_for_class(name_l: str) -> str:
    return LABEL_CLASS_MODE.get(name_l, LABEL_GLOBAL_MODE)

def build_label_text(name_l: str, score: float) -> str:
    mode = get_label_mode_for_class(name_l)
    if mode == 'none':
        return ''
    s = f'{score:.{SCORE_DECIMALS}f}'
    if mode == 'score_only':
        return s
    return f'{name_l} {s}'

def pick_color_bgr(cls_id: int, name: str):
    name_l = name.lower() if isinstance(name, str) else str(name).lower()
    if name_l in CUSTOM_COLORS_BGR:
        return CUSTOM_COLORS_BGR[name_l]
    # 默认 fallback：用 deadmood 的颜色
    return CUSTOM_COLORS_BGR['deadmood']


def xywh2xyxy(x):
    """[x, y, w, h] -> [x1, y1, x2, y2]"""
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def box_iou(box1, box2):
    """计算 IoU，box1: [N,4], box2: [M,4]，都是 xyxy"""
    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union = area1[:, None] + area2[None, :] - inter_area + 1e-7
    return inter_area / union


def nms(boxes, scores, iou_thres=0.5):
    """最简单版 NMS"""
    keep = []
    if boxes.numel() == 0:
        return keep

    idxs = scores.argsort(descending=True)
    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i.item())
        if idxs.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious <= iou_thres]
    return keep


def draw_and_save(img0, boxes, scores, labels, names, out_path):
    """
    画框 + 保存图片
    颜色、字体大小、文字颜色与你之前分析的脚本保持一致：
      brick_loss -> 青色，黑字
      vegetation -> 蓝色，白字，文字向左多移 2 个字符宽
      deadmood  -> 洋红/紫色，黑字
    文本内容：默认 '类名 分数'
    """
    img = img0.copy()
    H, W = img.shape[:2]

    # 线宽 & 字体大小：随图像尺寸自适应
    lw = max(round((H + W) / 2 * 0.003), 2)  # 线宽
    tf = max(lw - 1, 1)                      # 字体粗细
    fs = lw / 3                              # 字体大小
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 估算 SHIFT_CHARS 个字符宽度，用于 vegetation 左移
    ref_text = "0" * max(1, int(SHIFT_CHARS))
    (shift_width, _), _ = cv2.getTextSize(ref_text, font, fs, tf)

    # names 可能是 list 或 dict，这里统一处理
    if isinstance(names, dict):
        id2name = {int(k): v for k, v in names.items()}
    else:
        id2name = {i: n for i, n in enumerate(list(names))}

    for box, score, cls in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.int().tolist()
        cls_id = int(cls)
        name = id2name.get(cls_id, str(cls_id))
        name_l = name.lower() if isinstance(name, str) else str(name).lower()

        # 框颜色
        color = pick_color_bgr(cls_id, name_l)
        # 文字颜色：brick_loss/deadmood 用黑字，其他用白字
        txt_color = (0, 0, 0) if name_l in {'brick_loss', 'deadmood'} else (255, 255, 255)

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=lw)

        # 文本内容（可能为空：'none' 模式）
        label = build_label_text(name_l, float(score))
        if not label:
            continue

        # 计算文字尺寸
        (tw, th), base = cv2.getTextSize(label, font, fs, tf)
        y_text = max(y1, th + base + 3)

        x_text = x1 + 1
        # vegetation 左移 SHIFT_CHARS 个字符宽
        if name_l == SHIFT_CLASS:
            x_text = max(x_text - shift_width, 0)

        # 背景框
        bg_left  = max(x_text - 1, 0)
        bg_right = min(x_text + tw + 1, W - 1)
        cv2.rectangle(img,
                      (bg_left, y_text - th - base),
                      (bg_right, y_text),
                      color, -1)
        # 写文字
        cv2.putText(img, label, (x_text, y_text - base),
                    font, fs, txt_color, thickness=tf, lineType=cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    print("saved:", out_path)


def get_image_list(img_path):
    """既支持文件夹，也支持单张图片"""
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    if os.path.isdir(img_path):
        files = []
        for name in sorted(os.listdir(img_path)):
            full = os.path.join(img_path, name)
            if os.path.isfile(full) and name.lower().endswith(exts):
                files.append(full)
        return files
    elif os.path.isfile(img_path):
        return [img_path]
    else:
        raise FileNotFoundError(f"img_path 不存在：{img_path}")


def main():
    # ========= 1. 路径 & 参数 =========
    model_path = r'/home/xgq/Desktop/yolo/runs/train/exp25/weights/best.pt'  # 模型权重
    img_path = r'/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/dataset/6.2/images/val/(1795).jpg'  # 文件夹或单图
    # img_path = r'/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/dataset/6.2/images/val' 
    imgsz = 640  # 训练时尺寸

    # ========= 2. 加载模型 =========
    model = YOLO(model_path)
    det_model = model.model
    det_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det_model.to(device)

    # 检测头
    head = det_model.model[-1]
    if hasattr(head, "return_per_level"):
        head.return_per_level = True
        print("设置 head.return_per_level = True")
    else:
        print("head 没有 return_per_level 属性，脚本会尝试用通用方式拆分三个尺度。")

    print("最后一层类型:", type(head), head.__class__.__name__)
    names = model.names
    conf_thres = 0.15
    iou_thres = 0.1
    scale_names = ["p3", "p4", "p5"]

    # ========= 3. 处理图片列表 =========
    img_list = get_image_list(img_path)
    print(f"共找到 {len(img_list)} 张图片")

    for img_file in img_list:
        print(f"\n=== 处理: {img_file} ===")
        img0 = cv2.imread(img_file)
        assert img0 is not None, f"Image not found: {img_file}"
        h0, w0 = img0.shape[:2]

        # 简单 resize（如果要和训练完全一致，可以改成 letterbox）
        img = cv2.resize(img0, (imgsz, imgsz))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # ========= 4. 前向 =========
        with torch.no_grad():
            out = det_model(img_tensor)

        print("det_model 输出类型:", type(out))

        if not isinstance(out, tuple) or len(out) < 1:
            raise RuntimeError(f"det_model(img) 的返回类型异常: {type(out)}, 内容: {out}")

        first = out[0]
        preds_per_level = None

        # 情况 A：head 已经直接返回 (y_p3,y_p4,y_p5)
        if isinstance(first, (list, tuple)):
            preds_per_level = first
            print("检测到 head 已直接返回按尺度拆分结果, len(preds_per_level) =", len(preds_per_level))

        # 情况 B：返回 (y, feats)，用 feats 的 HW 拆分
        elif isinstance(first, torch.Tensor):
            if len(out) < 2:
                raise RuntimeError("out[0] 是 Tensor，但 out 里没有第二个元素用于辅助拆分(通常是中间特征 x)。")

            y = out[0]     # [B, C, sum(HW)]
            feats = out[1] # list/tuple，中间特征

            if not isinstance(feats, (list, tuple)):
                raise RuntimeError(f"out[1] 不是 list/tuple，而是 {type(feats)}，无法按尺度拆分。")

            hw = []
            for i, feat in enumerate(feats):
                if not hasattr(feat, "shape") or feat.dim() != 4:
                    raise RuntimeError(f"feats[{i}] 形状异常: {getattr(feat, 'shape', None)}")
                h, w = feat.shape[2], feat.shape[3]
                hw.append(h * w)
            print("通过中间特征自动拆分三个尺度，HW 列表:", hw)

            preds_per_level = torch.split(y, hw, dim=2)
            print("拆分后 preds_per_level 长度:", len(preds_per_level))

        else:
            raise RuntimeError(f"out[0] 的类型无法识别: {type(first)}")

        if preds_per_level is None:
            raise RuntimeError("无法得到 preds_per_level，请检查模型输出结构。")

        # ========= 5. 分别处理 P3 / P4 / P5 =========
        base_name = os.path.splitext(os.path.basename(img_file))[0]

        for idx, (y_level, sname) in enumerate(zip(preds_per_level, scale_names)):
            if not isinstance(y_level, torch.Tensor):
                print(f"{sname}: 该尺度输出不是 Tensor，而是 {type(y_level)}，跳过。")
                continue

            print(f"{sname} 原始输出形状:", y_level.shape)

            # 期望形状是 [B, 4+nc, HW]
            if y_level.dim() == 3:
                y_level = y_level[0].permute(1, 0)  # -> [HW, C]
            elif y_level.dim() == 2:
                y_level = y_level.permute(1, 0)     # -> [HW, C]
            else:
                print(f"{sname}: 维度 {y_level.dim()} 暂不支持，跳过。")
                continue

            print(f"{sname} 变换后形状:", y_level.shape)

            boxes_xywh = y_level[:, :4]
            cls_scores = y_level[:, 4:]  # [HW, nc]

            cls_conf, cls_idx = cls_scores.max(dim=1)

            mask = cls_conf > conf_thres
            if mask.sum() == 0:
                print(f"{sname}: no boxes over conf {conf_thres}")
                continue

            boxes_xywh = boxes_xywh[mask]
            cls_conf_ = cls_conf[mask]
            cls_idx_ = cls_idx[mask]

            boxes_xyxy = xywh2xyxy(boxes_xywh)

            gain_w = w0 / float(imgsz)
            gain_h = h0 / float(imgsz)
            boxes_xyxy[:, [0, 2]] *= gain_w
            boxes_xyxy[:, [1, 3]] *= gain_h

            keep = nms(boxes_xyxy, cls_conf_, iou_thres=iou_thres)
            if len(keep) == 0:
                print(f"{sname}: all boxes removed by NMS")
                continue

            boxes_keep = boxes_xyxy[keep].cpu()
            scores_keep = cls_conf_[keep].cpu()
            labels_keep = cls_idx_[keep].cpu()

            out_path = os.path.join("runs", "three_heads", f"{base_name}_{sname}.jpg")
            draw_and_save(img0, boxes_keep, scores_keep, labels_keep, names, out_path)


if __name__ == "__main__":
    main()
