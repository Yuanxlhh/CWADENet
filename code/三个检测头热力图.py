import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import os
import shutil
import time

import cv2
import torch
import numpy as np
from tqdm import trange
from PIL import Image

from ultralytics import YOLO
from pytorch_grad_cam import (
    GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM,
    HiResCAM, LayerCAM, RandomCAM, EigenGradCAM,
    KPCA_CAM, AblationCAM
)
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients as CamActivationsAndGradients

# ====================== 文字 & 颜色风格配置（与预测脚本保持一致） ======================

CUSTOM_COLORS_RGB = {
    'brick_loss': (0, 255, 255),   # 青色
    'vegetation': (0, 0, 255),     # 蓝色
    'deadmood':  (255, 0, 255),    # 洋红/紫
}
CUSTOM_COLORS_BGR = {k: (v[2], v[1], v[0]) for k, v in CUSTOM_COLORS_RGB.items()}

SHIFT_CLASS = 'vegetation'  # 只移动植被类
SHIFT_CHARS = 2             # 向左移动 2 个字符宽

# 全局：只显示分数，不显示病害名称
# 若想完全不显示文字可改成 'none'
LABEL_GLOBAL_MODE = 'none'
LABEL_CLASS_MODE = {
    # 可以按类单独覆盖：
    # 'brick_loss': 'none',
    # 'vegetation': 'score_only',
}
SCORE_DECIMALS = 2

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
    return CUSTOM_COLORS_BGR['deadmood']

# ====================== NMS / 坐标变换 ======================

def xywh2xyxy(x):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def box_iou(box1, box2):
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

# ====================== YOLO ActivationsAndGradients ======================

class ActivationsAndGradients:
    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))
            self.handles.append(target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients
        output.register_hook(_store_grad)

    def post_process(self, result):
        if self.model.end2end:
            logits_ = result[:, :, 4:]
            boxes_ = result[:, :, :4]
            sorted, indices = torch.sort(logits_[:, :, 0], descending=True)
            return logits_[0][indices[0]], boxes_[0][indices[0]]
        elif self.model.task == 'detect':
            logits_ = result[:, 4:]
            boxes_ = result[:, :4]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], 0, 1)[indices[0]], \
                   torch.transpose(boxes_[0], 0, 1)[indices[0]]
        elif self.model.task == 'segment':
            logits_ = result[0][:, 4:4 + self.model.nc]
            boxes_ = result[0][:, :4]
            mask_p, mask_nm = result[1][2].squeeze(), result[1][1].squeeze().transpose(1, 0)
            c, h, w = mask_p.size()
            mask = (mask_nm @ mask_p.view(c, -1))
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], 0, 1)[indices[0]], \
                   torch.transpose(boxes_[0], 0, 1)[indices[0]], \
                   mask[indices[0]]
        elif self.model.task == 'pose':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            poses_ = result[:, 4 + self.model.nc:]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], 0, 1)[indices[0]], \
                   torch.transpose(boxes_[0], 0, 1)[indices[0]], \
                   torch.transpose(poses_[0], 0, 1)[indices[0]]
        elif self.model.task == 'obb':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            angles_ = result[:, 4 + self.model.nc:]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], 0, 1)[indices[0]], \
                   torch.transpose(boxes_[0], 0, 1)[indices[0]], \
                   torch.transpose(angles_[0], 0, 1)[indices[0]]
        elif self.model.task == 'classify':
            return result[0]

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        if self.model.task == 'detect':
            post_result, pre_post_boxes = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes]]
        elif self.model.task == 'segment':
            post_result, pre_post_boxes, pre_post_mask = self.post_process(model_output)
            return [[post_result, pre_post_boxes, pre_post_mask]]
        elif self.model.task == 'pose':
            post_result, pre_post_boxes, pre_post_pose = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes, pre_post_pose]]
        elif self.model.task == 'obb':
            post_result, pre_post_boxes, pre_post_angle = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes, pre_post_angle]]
        elif self.model.task == 'classify':
            data = self.post_process(model_output)
            return [data]

    def release(self):
        for handle in self.handles:
            handle.remove()

# ============================== 目标函数 ==============================
class yolo_detect_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio, end2end) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
        self.end2end = end2end
    
    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if (self.end2end and float(post_result[i, 0]) < self.conf) or \
               (not self.end2end and float(post_result[i].max()) < self.conf):
                break
            if self.ouput_type in ['class', 'all']:
                result.append(post_result[i, 0] if self.end2end else post_result[i].max())
            if self.ouput_type in ['box', 'all']:
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)

class yolo_segment_target(yolo_detect_target):
    def forward(self, data):
        post_result, pre_post_boxes, pre_post_mask = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type in ['class', 'all']:
                result.append(post_result[i].max())
            if self.ouput_type in ['box', 'all']:
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            if self.ouput_type in ['segment', 'all']:
                result.append(pre_post_mask[i].mean())
        return sum(result)

class yolo_pose_target(yolo_detect_target):
    def forward(self, data):
        post_result, pre_post_boxes, pre_post_pose = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type in ['class', 'all']:
                result.append(post_result[i].max())
            if self.ouput_type in ['box', 'all']:
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            if self.ouput_type in ['pose', 'all']:
                result.append(pre_post_pose[i].mean())
        return sum(result)

class yolo_obb_target(yolo_detect_target):
    def forward(self, data):
        post_result, pre_post_boxes, pre_post_angle = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type in ['class', 'all']:
                result.append(post_result[i].max())
            if self.ouput_type in ['box', 'all']:
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            if self.ouput_type in ['obb', 'all']:
                result.append(pre_post_angle[i])
        return sum(result)

class yolo_classify_target(yolo_detect_target):
    def forward(self, data):
        return data.max()

# ============================== Grad-CAM 主类 ==============================
class yolo_heatmap:
    def __init__(self, weight, device, method, layer,
                 backward_type, conf_threshold, ratio,
                 show_result, renormalize, task, img_size):
        device = torch.device(device)
        self.device = device
        self.img_size = img_size
        self.task = task
        self.conf_threshold = conf_threshold
        self.ratio = ratio
        self.show_result = show_result
        self.renormalize = renormalize
        self.iou_thres = 0.1  # 与三头预测脚本一致

        model_yolo = YOLO(weight)
        print(f'model class info:{model_yolo.names}')
        model = model_yolo.model
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(True)

        model.task = task
        if not hasattr(model, 'end2end'):
            model.end2end = False

        self.model_yolo = model_yolo
        self.model = model
        self.det_model = model

        if isinstance(model_yolo.names, dict):
            id2name = {int(k): v for k, v in model_yolo.names.items()}
        else:
            id2name = {i: n for i, n in enumerate(model_yolo.names)}
        self.id2name = id2name

        if isinstance(conf_threshold, dict):
            base_conf = float(min(conf_threshold.values()))
        else:
            base_conf = float(conf_threshold)
        self.base_conf = base_conf

        if task == 'detect':
            target = yolo_detect_target(backward_type, base_conf, ratio, model.end2end)
        elif task == 'segment':
            target = yolo_segment_target(backward_type, base_conf, ratio, model.end2end)
        elif task == 'pose':
            target = yolo_pose_target(backward_type, base_conf, ratio, model.end2end)
        elif task == 'obb':
            target = yolo_obb_target(backward_type, base_conf, ratio, model.end2end)
        elif task == 'classify':
            target = yolo_classify_target(backward_type, base_conf, ratio, model.end2end)
        else:
            raise Exception(f"not support task({task}).")
        self.target = target

        method_name = method
        if layer and isinstance(layer[0], (list, tuple)):
            cam_methods = []
            for group in layer:
                tl = [model.model[l] for l in group]
                cam = eval(method_name)(model, tl)
                cam.activations_and_grads = ActivationsAndGradients(model, tl, None)
                cam_methods.append(cam)
            self.cam_methods = cam_methods
            self.method = cam_methods[0]
        else:
            tl = [model.model[l] for l in layer]
            cam = eval(method_name)(model, tl)
            cam.activations_and_grads = ActivationsAndGradients(model, tl, None)
            self.cam_methods = None
            self.method = cam

        self.color_map_bgr = {
            'vegetation': (0, 0, 255),
            'brick_loss': (0, 255, 255),
            'deadmood': (255, 0, 255)
        }

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(int(x1), 0), max(int(y1), 0)
            x2, y2 = min(int(x2), grayscale_cam.shape[1] - 1), min(int(y2), grayscale_cam.shape[0] - 1)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(
                grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np,
                                                        renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    # ========= 生成单尺度 CAM 图，最后 resize 回原图尺寸 =========
    def _render_cam_image(self, cam_method, tensor, img_rgb_float,
                          boxes_net, boxes_orig, cls_orig, conf_orig,
                          orig_h, orig_w):
        try:
            grayscale_cam = cam_method(tensor, [self.target])
        except AttributeError:
            print("Warning... cam_method(tensor, [self.target]) failure.")
            return None

        grayscale_cam = grayscale_cam[0, :]

        if self.renormalize and self.task in ['detect', 'segment', 'pose'] \
           and boxes_net is not None and len(boxes_net) > 0:
            cam_image_rgb = self.renormalize_cam_in_bounding_boxes(
                boxes_net, img_rgb_float, grayscale_cam
            )
        else:
            cam_image_rgb = show_cam_on_image(img_rgb_float, grayscale_cam, use_rgb=True)

        cam_image_bgr = cv2.cvtColor(cam_image_rgb, cv2.COLOR_RGB2BGR)

        cam_image = cv2.resize(cam_image_bgr, (orig_w, orig_h))

        if self.show_result and boxes_orig is not None and len(boxes_orig) > 0:
            H, W = cam_image.shape[:2]
            lw = max(round((H + W) / 2 * 0.003), 2)
            tf = max(lw - 1, 1)
            fs = lw / 3
            font = cv2.FONT_HERSHEY_SIMPLEX

            ref_text = "0" * max(1, int(SHIFT_CHARS))
            (shift_width, _), _ = cv2.getTextSize(ref_text, font, fs, tf)

            for i, (box, c) in enumerate(zip(boxes_orig, cls_orig)):
                x1, y1, x2, y2 = map(int, box)
                cls_id = int(c)
                name = self.id2name.get(cls_id, f'class{cls_id}')
                name_l = name.lower() if isinstance(name, str) else str(name).lower()

                color_box = pick_color_bgr(cls_id, name_l)
                txt_color = (0, 0, 0) if name_l in {'brick_loss', 'deadmood'} else (255, 255, 255)

                score = float(conf_orig[i]) if conf_orig is not None else 1.0
                label = build_label_text(name_l, score)  # 这里现在是“只显示分数”

                cv2.rectangle(cam_image, (x1, y1), (x2, y2), color_box, lw)

                if not label:
                    continue

                (tw, th), base = cv2.getTextSize(label, font, fs, tf)
                y_text = max(y1, th + base + 3)
                x_text = x1 + 1
                if name_l == SHIFT_CLASS:
                    x_text = max(x_text - shift_width, 0)

                bg_left  = max(x_text - 1, 0)
                bg_right = min(x_text + tw + 1, W - 1)
                cv2.rectangle(
                    cam_image,
                    (bg_left, y_text - th - base),
                    (bg_right, y_text),
                    color_box, -1
                )
                cv2.putText(cam_image, label, (x_text, y_text - base),
                            font, fs, txt_color, thickness=tf, lineType=cv2.LINE_AA)

        cam_image_rgb_out = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
        cam_image_pil = Image.fromarray(cam_image_rgb_out)
        return cam_image_pil


    # ========= 主处理：按 P3/P4/P5 出图 =========
    def process(self, img_path, save_base_path):
        try:
            img0 = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
        except:
            print(f"Warning... {img_path} read failure.")
            return
        if img0 is None:
            print(f"Warning... {img_path} read failure (None).")
            return

        h0, w0 = img0.shape[:2]

        img = cv2.resize(img0, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb_float = np.float32(img_rgb) / 255.0
        tensor = torch.from_numpy(
            np.transpose(img_rgb_float, (2, 0, 1))
        ).unsqueeze(0).to(self.device)
        print(f'tensor size:{tensor.size()}')

        with torch.no_grad():
            out = self.det_model(tensor)

        print("det_model 输出类型:", type(out))
        if not isinstance(out, tuple) or len(out) < 1:
            raise RuntimeError(f"det_model(img) 的返回类型异常: {type(out)}, 内容: {out}")

        first = out[0]
        preds_per_level = None

        if isinstance(first, (list, tuple)):
            preds_per_level = first
            print("检测到 head 已直接返回按尺度拆分结果, len(preds_per_level) =", len(preds_per_level))
        elif isinstance(first, torch.Tensor):
            if len(out) < 2:
                raise RuntimeError("out[0] 是 Tensor，但 out 里没有第二个元素用于辅助拆分(通常是中间特征 x)。")
            y = out[0]
            feats = out[1]
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

        scale_names = ["p3", "p4", "p5"]
        base, ext = os.path.splitext(save_base_path)
        if ext == '':
            ext = '.png'

        gain_w = w0 / float(self.img_size)
        gain_h = h0 / float(self.img_size)

        for idx, (y_level, sname) in enumerate(zip(preds_per_level, scale_names)):
            if not isinstance(y_level, torch.Tensor):
                print(f"{sname}: 该尺度输出不是 Tensor，而是 {type(y_level)}，跳过。")
                continue

            print(f"{sname} 原始输出形状:", y_level.shape)

            if y_level.dim() == 3:
                y_level = y_level[0].permute(1, 0)
            elif y_level.dim() == 2:
                y_level = y_level.permute(1, 0)
            else:
                print(f"{sname}: 维度 {y_level.dim()} 暂不支持，跳过。")
                continue

            print(f"{sname} 变换后形状:", y_level.shape)

            boxes_xywh = y_level[:, :4]
            cls_scores = y_level[:, 4:]

            cls_conf, cls_idx = cls_scores.max(dim=1)

            mask = cls_conf > self.base_conf
            if mask.sum() == 0:
                print(f"{sname}: no boxes over conf {self.base_conf}")
                continue

            boxes_xywh = boxes_xywh[mask]
            cls_conf_ = cls_conf[mask]
            cls_idx_ = cls_idx[mask]

            boxes_xyxy_net = xywh2xyxy(boxes_xywh.clone())

            boxes_xyxy_orig = boxes_xyxy_net.clone()
            boxes_xyxy_orig[:, [0, 2]] *= gain_w
            boxes_xyxy_orig[:, [1, 3]] *= gain_h

            keep = nms(boxes_xyxy_orig, cls_conf_, iou_thres=self.iou_thres)
            if len(keep) == 0:
                print(f"{sname}: all boxes removed by NMS")
                continue

            boxes_keep_orig = boxes_xyxy_orig[keep].cpu().numpy()
            scores_keep = cls_conf_[keep].cpu().numpy()
            labels_keep = cls_idx_[keep].cpu().numpy()
            boxes_keep_net = boxes_xyxy_net[keep].cpu().numpy()

            if self.cam_methods is None:
                cam_m = self.method
            else:
                cam_m = self.cam_methods[min(idx, len(self.cam_methods)-1)]

            out_path = f"{base}_{sname}{ext}"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cam_image = self._render_cam_image(
                cam_m, tensor, img_rgb_float,
                boxes_keep_net, boxes_keep_orig, labels_keep, scores_keep,
                h0, w0
            )
            if cam_image is not None:
                cam_image.save(out_path)
                print("saved:", out_path)

    def __call__(self, img_path, save_root):
        if os.path.exists(save_root):
            shutil.rmtree(save_root)
        os.makedirs(save_root, exist_ok=True)

        img_list = get_image_list(img_path)
        print(f"共找到 {len(img_list)} 张图片用于 Grad-CAM 可视化")
        for p in img_list:
            base = os.path.splitext(os.path.basename(p))[0]
            save_path = os.path.join(save_root, base + '.png')
            self.process(p, save_path)

# ============================== 获取图片列表 ==============================
def get_image_list(img_path):
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

# ============================== 参数 ==============================
def get_params():
    params = {
        'weight':  r'/home/xgq/Desktop/yolo/runs/train/exp25/weights/best.pt',
        'device': 'cuda:0',
        'method': 'GradCAMPlusPlus',
        'layer': [
            [28],  # P3
            [31],  # P4
            [34]       # P5
        ],
        'backward_type': 'all',
        'conf_threshold': 0.15,
        'ratio': 0.02,
        'show_result': True,
        'renormalize': False,
        'task': 'detect',
        'img_size': 640,
    }
    return params

# ============================== 入口 ==============================
if __name__ == '__main__':
    save_dir = time.strftime("result_%Y%m%d_%H%M%S")
    model = yolo_heatmap(**get_params())
    # 文件夹或单图都可以：
    model(r'/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/dataset/6.2/images/val/(1717).jpg',
          save_dir)
