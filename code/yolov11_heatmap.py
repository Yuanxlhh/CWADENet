# import warnings
# warnings.filterwarnings('ignore')
# warnings.simplefilter('ignore')
# import torch, yaml, cv2, os, shutil, sys, copy
# import numpy as np
# np.random.seed(0)
# import matplotlib.pyplot as plt
# from tqdm import trange
# from PIL import Image
# from ultralytics import YOLO
# from ultralytics.nn.tasks import attempt_load_weights
# from ultralytics.utils.torch_utils import intersect_dicts
# from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
# from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, KPCA_CAM, AblationCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
# from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

# def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
#     # Resize and pad image while meeting stride-multiple constraints
#     shape = im.shape[:2]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)

#     # Scale ratio (new / old)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better val mAP)
#         r = min(r, 1.0)

#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

#     dw /= 2  # divide padding into 2 sides
#     dh /= 2

#     if shape[::-1] != new_unpad:  # resize
#         im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#     return im, ratio, (top, bottom, left, right)

# class ActivationsAndGradients:
#     """ Class for extracting activations and
#     registering gradients from targetted intermediate layers """

#     def __init__(self, model, target_layers, reshape_transform):
#         self.model = model
#         self.gradients = []
#         self.activations = []
#         self.reshape_transform = reshape_transform
#         self.handles = []
#         for target_layer in target_layers:
#             self.handles.append(
#                 target_layer.register_forward_hook(self.save_activation))
#             # Because of https://github.com/pytorch/pytorch/issues/61519,
#             # we don't use backward hook to record gradients.
#             self.handles.append(
#                 target_layer.register_forward_hook(self.save_gradient))

#     def save_activation(self, module, input, output):
#         activation = output

#         if self.reshape_transform is not None:
#             activation = self.reshape_transform(activation)
#         self.activations.append(activation.cpu().detach())

#     def save_gradient(self, module, input, output):
#         if not hasattr(output, "requires_grad") or not output.requires_grad:
#             # You can only register hooks on tensor requires grad.
#             return

#         # Gradients are computed in reverse order
#         def _store_grad(grad):
#             if self.reshape_transform is not None:
#                 grad = self.reshape_transform(grad)
#             self.gradients = [grad.cpu().detach()] + self.gradients

#         output.register_hook(_store_grad)

#     def post_process(self, result):
#         if self.model.end2end:
#             logits_ = result[:, :, 4:]
#             boxes_ = result[:, :, :4]
#             sorted, indices = torch.sort(logits_[:, :, 0], descending=True)
#             return logits_[0][indices[0]], boxes_[0][indices[0]]
#         elif self.model.task == 'detect':
#             logits_ = result[:, 4:]
#             boxes_ = result[:, :4]
#             sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
#             return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]
#         elif self.model.task == 'segment':
#             logits_ = result[0][:, 4:4 + self.model.nc]
#             boxes_ = result[0][:, :4]
#             mask_p, mask_nm = result[1][2].squeeze(), result[1][1].squeeze().transpose(1, 0)
#             c, h, w = mask_p.size()
#             mask = (mask_nm @ mask_p.view(c, -1))
#             sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
#             return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], mask[indices[0]]
#         elif self.model.task == 'pose':
#             logits_ = result[:, 4:4 + self.model.nc]
#             boxes_ = result[:, :4]
#             poses_ = result[:, 4 + self.model.nc:]
#             sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
#             return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(poses_[0], dim0=0, dim1=1)[indices[0]]
#         elif self.model.task == 'obb':
#             logits_ = result[:, 4:4 + self.model.nc]
#             boxes_ = result[:, :4]
#             angles_ = result[:, 4 + self.model.nc:]
#             sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
#             return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(angles_[0], dim0=0, dim1=1)[indices[0]]
#         elif self.model.task == 'classify':
#             return result[0]
  
#     def __call__(self, x):
#         self.gradients = []
#         self.activations = []
#         model_output = self.model(x)
#         if self.model.task == 'detect':
#             post_result, pre_post_boxes = self.post_process(model_output[0])
#             return [[post_result, pre_post_boxes]]
#         elif self.model.task == 'segment':
#             post_result, pre_post_boxes, pre_post_mask = self.post_process(model_output)
#             return [[post_result, pre_post_boxes, pre_post_mask]]
#         elif self.model.task == 'pose':
#             post_result, pre_post_boxes, pre_post_pose = self.post_process(model_output[0])
#             return [[post_result, pre_post_boxes, pre_post_pose]]
#         elif self.model.task == 'obb':
#             post_result, pre_post_boxes, pre_post_angle = self.post_process(model_output[0])
#             return [[post_result, pre_post_boxes, pre_post_angle]]
#         elif self.model.task == 'classify':
#             data = self.post_process(model_output)
#             return [data]

#     def release(self):
#         for handle in self.handles:
#             handle.remove()

# class yolo_detect_target(torch.nn.Module):
#     def __init__(self, ouput_type, conf, ratio, end2end) -> None:
#         super().__init__()
#         self.ouput_type = ouput_type
#         self.conf = conf
#         self.ratio = ratio
#         self.end2end = end2end
    
#     def forward(self, data):
#         post_result, pre_post_boxes = data
#         result = []
#         for i in trange(int(post_result.size(0) * self.ratio)):
#             if (self.end2end and float(post_result[i, 0]) < self.conf) or (not self.end2end and float(post_result[i].max()) < self.conf):
#                 break
#             if self.ouput_type == 'class' or self.ouput_type == 'all':
#                 if self.end2end:
#                     result.append(post_result[i, 0])
#                 else:
#                     result.append(post_result[i].max())
#             elif self.ouput_type == 'box' or self.ouput_type == 'all':
#                 for j in range(4):
#                     result.append(pre_post_boxes[i, j])
#         return sum(result)

# class yolo_segment_target(yolo_detect_target):
#     def __init__(self, ouput_type, conf, ratio, end2end):
#         super().__init__(ouput_type, conf, ratio, end2end)
    
#     def forward(self, data):
#         post_result, pre_post_boxes, pre_post_mask = data
#         result = []
#         for i in trange(int(post_result.size(0) * self.ratio)):
#             if float(post_result[i].max()) < self.conf:
#                 break
#             if self.ouput_type == 'class' or self.ouput_type == 'all':
#                 result.append(post_result[i].max())
#             elif self.ouput_type == 'box' or self.ouput_type == 'all':
#                 for j in range(4):
#                     result.append(pre_post_boxes[i, j])
#             elif self.ouput_type == 'segment' or self.ouput_type == 'all':
#                 result.append(pre_post_mask[i].mean())
#         return sum(result)

# class yolo_pose_target(yolo_detect_target):
#     def __init__(self, ouput_type, conf, ratio, end2end):
#         super().__init__(ouput_type, conf, ratio, end2end)
    
#     def forward(self, data):
#         post_result, pre_post_boxes, pre_post_pose = data
#         result = []
#         for i in trange(int(post_result.size(0) * self.ratio)):
#             if float(post_result[i].max()) < self.conf:
#                 break
#             if self.ouput_type == 'class' or self.ouput_type == 'all':
#                 result.append(post_result[i].max())
#             elif self.ouput_type == 'box' or self.ouput_type == 'all':
#                 for j in range(4):
#                     result.append(pre_post_boxes[i, j])
#             elif self.ouput_type == 'pose' or self.ouput_type == 'all':
#                 result.append(pre_post_pose[i].mean())
#         return sum(result)

# class yolo_obb_target(yolo_detect_target):
#     def __init__(self, ouput_type, conf, ratio, end2end):
#         super().__init__(ouput_type, conf, ratio, end2end)
    
#     def forward(self, data):
#         post_result, pre_post_boxes, pre_post_angle = data
#         result = []
#         for i in trange(int(post_result.size(0) * self.ratio)):
#             if float(post_result[i].max()) < self.conf:
#                 break
#             if self.ouput_type == 'class' or self.ouput_type == 'all':
#                 result.append(post_result[i].max())
#             elif self.ouput_type == 'box' or self.ouput_type == 'all':
#                 for j in range(4):
#                     result.append(pre_post_boxes[i, j])
#             elif self.ouput_type == 'obb' or self.ouput_type == 'all':
#                 result.append(pre_post_angle[i])
#         return sum(result)

# class yolo_classify_target(yolo_detect_target):
#     def __init__(self, ouput_type, conf, ratio, end2end):
#         super().__init__(ouput_type, conf, ratio, end2end)
    
#     def forward(self, data):
#         return data.max()

# class yolo_heatmap:
#     def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_result, renormalize, task, img_size):
#         device = torch.device(device)
#         model_yolo = YOLO(weight)
#         model_names = model_yolo.names
#         print(f'model class info:{model_names}')
#         model = copy.deepcopy(model_yolo.model)
#         model.to(device)
#         model.info()
#         for p in model.parameters():
#             p.requires_grad_(True)
#         model.eval()
        
#         model.task = task
#         if not hasattr(model, 'end2end'):
#             model.end2end = False
        
#         if task == 'detect':
#             target = yolo_detect_target(backward_type, conf_threshold, ratio, model.end2end)
#         elif task == 'segment':
#             target = yolo_segment_target(backward_type, conf_threshold, ratio, model.end2end)
#         elif task == 'pose':
#             target = yolo_pose_target(backward_type, conf_threshold, ratio, model.end2end)
#         elif task == 'obb':
#             target = yolo_obb_target(backward_type, conf_threshold, ratio, model.end2end)
#         elif task == 'classify':
#             target = yolo_classify_target(backward_type, conf_threshold, ratio, model.end2end)
#         else:
#             raise Exception(f"not support task({task}).")
        
#         target_layers = [model.model[l] for l in layer]
#         method = eval(method)(model, target_layers)
#         method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)
        
#         colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
#         self.__dict__.update(locals())
    
#     def post_process(self, result):
#         result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
#         return result

#     def draw_detections(self, box, color, name, img):
#         xmin, ymin, xmax, ymax = list(map(int, list(box)))
#         cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2) # 绘制检测框
#         cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2, lineType=cv2.LINE_AA)  # 绘制类别、置信度
#         return img

#     def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
#         """Normalize the CAM to be in the range [0, 1] 
#         inside every bounding boxes, and zero outside of the bounding boxes. """
#         renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
#         for x1, y1, x2, y2 in boxes:
#             x1, y1 = max(x1, 0), max(y1, 0)
#             x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
#             renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
#         renormalized_cam = scale_cam_image(renormalized_cam)
#         eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
#         return eigencam_image_renormalized
    
#     def process(self, img_path, save_path):
#         # img process
#         try:
#             img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
#         except:
#             print(f"Warning... {img_path} read failure.")
#             return
#         img, _, (top, bottom, left, right) = letterbox(img, new_shape=(self.img_size, self.img_size), auto=True) # 如果需要完全固定成宽高一样就把auto设置为False
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.float32(img) / 255.0
#         tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
#         print(f'tensor size:{tensor.size()}')
        
#         try:
#             grayscale_cam = self.method(tensor, [self.target])
#         except AttributeError as e:
#             print(f"Warning... self.method(tensor, [self.target]) failure.")
#             return
        
#         grayscale_cam = grayscale_cam[0, :]
#         cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        
#         pred = self.model_yolo.predict(tensor, conf=self.conf_threshold, iou=0.4)[0]
#         if self.renormalize and self.task in ['detect', 'segment', 'pose']:
#             cam_image = self.renormalize_cam_in_bounding_boxes(pred.boxes.xyxy.cpu().detach().numpy().astype(np.int32), img, grayscale_cam)
#         if self.show_result:
#             cam_image = pred.plot(img=cam_image,
#                                   conf=True, # 显示置信度
#                                   font_size=None, # 字体大小，None为根据当前image尺寸计算
#                                   line_width=None, # 线条宽度，None为根据当前image尺寸计算
#                                   labels=False, # 显示标签
#                                   )
        
#         # 去掉padding边界
#         cam_image = cam_image[top:cam_image.shape[0] - bottom, left:cam_image.shape[1] - right]
#         cam_image = Image.fromarray(cam_image)
#         cam_image.save(save_path)
    
#     def __call__(self, img_path, save_path):
#         # remove dir if exist
#         if os.path.exists(save_path):
#             shutil.rmtree(save_path)
#         # make dir if not exist
#         os.makedirs(save_path, exist_ok=True)

#         if os.path.isdir(img_path):
#             for img_path_ in os.listdir(img_path):
#                 self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}')
#         else:
#             self.process(img_path, f'{save_path}/result.png')
        
# def get_params():
#     params = {
#         'weight': r'/home/xgq/Desktop/yolo/runs/train/exp69/weights/best.pt', # 现在只需要指定权重即可,不需要指定cfg
#         'device': 'cuda:0',
#         'method': 'GradCAMPlusPlus', # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, KPCA_CAM
#         'layer': [10, 12, 14, 16, 18],
#         'backward_type': 'all', # detect:<class, box, all> segment:<class, box, segment, all> pose:<box, keypoint, all> obb:<box, angle, all> classify:<all>
#         'conf_threshold': 0.41, # 0.2
#         'ratio': 0.02, # 0.02-0.1
#         'show_result': True, # 不需要绘制结果请设置为False
#         'renormalize': False, # 需要把热力图限制在框内请设置为True(仅对detect,segment,pose有效)
#         'task':'detect', # 任务(detect,segment,pose,obb,classify)
#         'img_size':640, # 图像尺寸
#     }
#     return params

# # pip install grad-cam==1.5.4 --no-deps
# # if __name__ == '__main__':
# #     model = yolo_heatmap(**get_params())
#     # model(r'/home/hjj/Desktop/dataset/dataset_coco/coco/images/val2017/000000361238.jpg', 'result')
#     # model(r'/home/xgq/Desktop/ultralytics-yolo11-20250502/ultralytics-yolo11-main/dataset/6.2/images/val', 'result')
# import time

# if __name__ == '__main__':
#     save_dir = time.strftime("result_%Y%m%d_%H%M%S")  # 示例: result_20250710_214500
#     model = yolo_heatmap(**get_params())
#     # model(r'C:\Users\18084\Desktop\yolo\YOLO\ultralytics-yolo11-20250502\ultralytics-yolo11-main\dataset\6.2\images\val', save_dir)
#     model(r'/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/dataset/6.2/images/val/(1).jpg', save_dir)



# import warnings
# warnings.filterwarnings('ignore')
# warnings.simplefilter('ignore')
# import torch, yaml, cv2, os, shutil, sys, copy, time
# import numpy as np
# np.random.seed(0)
# import matplotlib.pyplot as plt
# from tqdm import trange
# from PIL import Image
# from ultralytics import YOLO
# from ultralytics.nn.tasks import attempt_load_weights
# from ultralytics.utils.torch_utils import intersect_dicts
# from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
# from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, KPCA_CAM, AblationCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
# from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

# def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
#     # Resize and pad image while meeting stride-multiple constraints
#     shape = im.shape[:2]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)

#     # Scale ratio (new / old)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better val mAP)
#         r = min(r, 1.0)

#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

#     dw /= 2  # divide padding into 2 sides
#     dh /= 2

#     if shape[::-1] != new_unpad:  # resize
#         im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#     return im, ratio, (top, bottom, left, right)

# class ActivationsAndGradients:
#     """ Class for extracting activations and registering gradients from targetted intermediate layers """

#     def __init__(self, model, target_layers, reshape_transform):
#         self.model = model
#         self.gradients = []
#         self.activations = []
#         self.reshape_transform = reshape_transform
#         self.handles = []
#         for target_layer in target_layers:
#             self.handles.append(target_layer.register_forward_hook(self.save_activation))
#             # Because of https://github.com/pytorch/pytorch/issues/61519, we don't use backward hook to record gradients.
#             self.handles.append(target_layer.register_forward_hook(self.save_gradient))

#     def save_activation(self, module, input, output):
#         activation = output
#         if self.reshape_transform is not None:
#             activation = self.reshape_transform(activation)
#         self.activations.append(activation.cpu().detach())

#     def save_gradient(self, module, input, output):
#         if not hasattr(output, "requires_grad") or not output.requires_grad:
#             return
#         def _store_grad(grad):
#             if self.reshape_transform is not None:
#                 grad = self.reshape_transform(grad)
#             self.gradients = [grad.cpu().detach()] + self.gradients
#         output.register_hook(_store_grad)

#     def post_process(self, result):
#         if self.model.end2end:
#             logits_ = result[:, :, 4:]
#             boxes_ = result[:, :, :4]
#             sorted, indices = torch.sort(logits_[:, :, 0], descending=True)
#             return logits_[0][indices[0]], boxes_[0][indices[0]]
#         elif self.model.task == 'detect':
#             logits_ = result[:, 4:]
#             boxes_ = result[:, :4]
#             sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
#             return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]
#         elif self.model.task == 'segment':
#             logits_ = result[0][:, 4:4 + self.model.nc]
#             boxes_ = result[0][:, :4]
#             mask_p, mask_nm = result[1][2].squeeze(), result[1][1].squeeze().transpose(1, 0)
#             c, h, w = mask_p.size()
#             mask = (mask_nm @ mask_p.view(c, -1))
#             sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
#             return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], mask[indices[0]]
#         elif self.model.task == 'pose':
#             logits_ = result[:, 4:4 + self.model.nc]
#             boxes_ = result[:, :4]
#             poses_ = result[:, 4 + self.model.nc:]
#             sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
#             return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(poses_[0], dim0=0, dim1=1)[indices[0]]
#         elif self.model.task == 'obb':
#             logits_ = result[:, 4:4 + self.model.nc]
#             boxes_ = result[:, :4]
#             angles_ = result[:, 4 + self.model.nc:]
#             sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
#             return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(angles_[0], dim0=0, dim1=1)[indices[0]]
#         elif self.model.task == 'classify':
#             return result[0]
  
#     def __call__(self, x):
#         self.gradients = []
#         self.activations = []
#         model_output = self.model(x)
#         if self.model.task == 'detect':
#             post_result, pre_post_boxes = self.post_process(model_output[0])
#             return [[post_result, pre_post_boxes]]
#         elif self.model.task == 'segment':
#             post_result, pre_post_boxes, pre_post_mask = self.post_process(model_output)
#             return [[post_result, pre_post_boxes, pre_post_mask]]
#         elif self.model.task == 'pose':
#             post_result, pre_post_boxes, pre_post_pose = self.post_process(model_output[0])
#             return [[post_result, pre_post_boxes, pre_post_pose]]
#         elif self.model.task == 'obb':
#             post_result, pre_post_boxes, pre_post_angle = self.post_process(model_output[0])
#             return [[post_result, pre_post_boxes, pre_post_angle]]
#         elif self.model.task == 'classify':
#             data = self.post_process(model_output)
#             return [data]

#     def release(self):
#         for handle in self.handles:
#             handle.remove()

# class yolo_detect_target(torch.nn.Module):
#     def __init__(self, ouput_type, conf, ratio, end2end) -> None:
#         super().__init__()
#         self.ouput_type = ouput_type
#         self.conf = conf
#         self.ratio = ratio
#         self.end2end = end2end
    
#     def forward(self, data):
#         post_result, pre_post_boxes = data
#         result = []
#         for i in trange(int(post_result.size(0) * self.ratio)):
#             if (self.end2end and float(post_result[i, 0]) < self.conf) or (not self.end2end and float(post_result[i].max()) < self.conf):
#                 break
#             if self.ouput_type == 'class' or self.ouput_type == 'all':
#                 if self.end2end:
#                     result.append(post_result[i, 0])
#                 else:
#                     result.append(post_result[i].max())
#             elif self.ouput_type == 'box' or self.ouput_type == 'all':
#                 for j in range(4):
#                     result.append(pre_post_boxes[i, j])
#         return sum(result)

# class yolo_segment_target(yolo_detect_target):
#     def __init__(self, ouput_type, conf, ratio, end2end):
#         super().__init__(ouput_type, conf, ratio, end2end)
    
#     def forward(self, data):
#         post_result, pre_post_boxes, pre_post_mask = data
#         result = []
#         for i in trange(int(post_result.size(0) * self.ratio)):
#             if float(post_result[i].max()) < self.conf:
#                 break
#             if self.ouput_type == 'class' or self.ouput_type == 'all':
#                 result.append(post_result[i].max())
#             elif self.ouput_type == 'box' or self.ouput_type == 'all':
#                 for j in range(4):
#                     result.append(pre_post_boxes[i, j])
#             elif self.ouput_type == 'segment' or self.ouput_type == 'all':
#                 result.append(pre_post_mask[i].mean())
#         return sum(result)

# class yolo_pose_target(yolo_detect_target):
#     def __init__(self, ouput_type, conf, ratio, end2end):
#         super().__init__(ouput_type, conf, ratio, end2end)
    
#     def forward(self, data):
#         post_result, pre_post_boxes, pre_post_pose = data
#         result = []
#         for i in trange(int(post_result.size(0) * self.ratio)):
#             if float(post_result[i].max()) < self.conf:
#                 break
#             if self.ouput_type == 'class' or self.ouput_type == 'all':
#                 result.append(post_result[i].max())
#             elif self.ouput_type == 'box' or self.ouput_type == 'all':
#                 for j in range(4):
#                     result.append(pre_post_boxes[i, j])
#             elif self.ouput_type == 'pose' or self.ouput_type == 'all':
#                 result.append(pre_post_pose[i].mean())
#         return sum(result)

# class yolo_obb_target(yolo_detect_target):
#     def __init__(self, ouput_type, conf, ratio, end2end):
#         super().__init__(ouput_type, conf, ratio, end2end)
    
#     def forward(self, data):
#         post_result, pre_post_boxes, pre_post_angle = data
#         result = []
#         for i in trange(int(post_result.size(0) * self.ratio)):
#             if float(post_result[i].max()) < self.conf:
#                 break
#             if self.ouput_type == 'class' or self.ouput_type == 'all':
#                 result.append(post_result[i].max())
#             elif self.ouput_type == 'box' or self.ouput_type == 'all':
#                 for j in range(4):
#                     result.append(pre_post_boxes[i, j])
#             elif self.ouput_type == 'obb' or self.ouput_type == 'all':
#                 result.append(pre_post_angle[i])
#         return sum(result)

# class yolo_classify_target(yolo_detect_target):
#     def __init__(self, ouput_type, conf, ratio, end2end):
#         super().__init__(ouput_type, conf, ratio, end2end)
    
#     def forward(self, data):
#         return data.max()

# class yolo_heatmap:
#     def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_result, renormalize, task, img_size):
#         device = torch.device(device)
#         model_yolo = YOLO(weight)
#         model_names = model_yolo.names
#         print(f'model class info:{model_names}')
#         model = copy.deepcopy(model_yolo.model)
#         model.to(device)
#         model.info()
#         for p in model.parameters():
#             p.requires_grad_(True)
#         model.eval()
        
#         model.task = task
#         if not hasattr(model, 'end2end'):
#             model.end2end = False
        
#         # ---- 仅此处改动：若传 dict，则用最小阈值供 CAM/初次预测取候选 ----
#         if isinstance(conf_threshold, dict):
#             conf_for_cam = float(min(conf_threshold.values()))
#         else:
#             conf_for_cam = float(conf_threshold)

#         if task == 'detect':
#             target = yolo_detect_target(backward_type, conf_for_cam, ratio, model.end2end)
#         elif task == 'segment':
#             target = yolo_segment_target(backward_type, conf_for_cam, ratio, model.end2end)
#         elif task == 'pose':
#             target = yolo_pose_target(backward_type, conf_for_cam, ratio, model.end2end)
#         elif task == 'obb':
#             target = yolo_obb_target(backward_type, conf_for_cam, ratio, model.end2end)
#         elif task == 'classify':
#             target = yolo_classify_target(backward_type, conf_for_cam, ratio, model.end2end)
#         else:
#             raise Exception(f"not support task({task}).")
        
#         target_layers = [model.model[l] for l in layer]
#         method = eval(method)(model, target_layers)
#         method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

#         # 自定义框颜色（BGR）：vegetation=蓝, brick_loss=青
#         self.color_map_bgr = {'vegetation': (0, 0, 255), 'brick_loss': (0, 255, 255),'deadmood': (255,0,255)}

#         self.__dict__.update(locals())
    
#     def post_process(self, result):
#         result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.05)[0]
#         return result

#     def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
#         renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
#         for x1, y1, x2, y2 in boxes:
#             x1, y1 = max(x1, 0), max(y1, 0)
#             x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
#             renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
#         renormalized_cam = scale_cam_image(renormalized_cam)
#         eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
#         return eigencam_image_renormalized
    
#     def process(self, img_path, save_path):
#         # img process（保持原样）
#         try:
#             img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
#         except:
#             print(f"Warning... {img_path} read failure.")
#             return
#         img, _, (top, bottom, left, right) = letterbox(img, new_shape=(self.img_size, self.img_size), auto=True)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.float32(img) / 255.0
#         tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
#         print(f'tensor size:{tensor.size()}')
        
#         # --- Grad-CAM（保持原来做法 & 叠色方式，确保热力图一致）---
#         try:
#             grayscale_cam = self.method(tensor, [self.target])
#         except AttributeError:
#             print(f"Warning... self.method(tensor, [self.target]) failure.")
#             return
#         grayscale_cam = grayscale_cam[0, :]
#         cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

#         # --- 预测：先用最小阈值拿候选 ---
#         if isinstance(self.conf_threshold, dict):
#             conf_for_predict = float(min(self.conf_threshold.values()))
#         else:
#             conf_for_predict = float(self.conf_threshold)
#         pred = self.model_yolo.predict(tensor, conf=conf_for_predict, iou=0.05)[0]

#         # 若无框，直接保存热力图
#         if pred.boxes is None or len(pred.boxes) == 0:
#             out = Image.fromarray(cam_image[top:cam_image.shape[0] - bottom, left:cam_image.shape[1] - right])
#             out.save(save_path)
#             return

#         # 读取 numpy
#         id2name = self.model_yolo.names if isinstance(self.model_yolo.names, dict) \
#                   else {i: n for i, n in enumerate(self.model_yolo.names)}
#         cls_np  = pred.boxes.cls.cpu().numpy().astype(int)
#         conf_np = pred.boxes.conf.cpu().numpy()
#         boxes   = pred.boxes.xyxy.cpu().numpy().astype(int)

#         # --- 按类别阈值二次过滤（不改 pred 原始对象，避免只读属性报错）---
#         if isinstance(self.conf_threshold, dict):
#             keep_mask = np.array([conf_np[i] >= float(self.conf_threshold.get(int(cls_np[i]), conf_for_predict))
#                                   for i in range(len(cls_np))], dtype=bool)
#         else:
#             keep_mask = conf_np >= float(self.conf_threshold)
#         boxes_f = boxes[keep_mask]
#         cls_f   = cls_np[keep_mask]

#         # 可选：把 CAM 限定在框内（和你原逻辑一致，默认 False）
#         if self.renormalize and self.task in ['detect', 'segment', 'pose'] and len(boxes_f) > 0:
#             cam_image = self.renormalize_cam_in_bounding_boxes(boxes_f, img, grayscale_cam)

#         # --- 自定义颜色画框（不画文字），保持与原风格类似的线宽自适应 ---
#         if self.show_result and len(boxes_f) > 0:
#             H, W = cam_image.shape[:2]
#             lw = max(round((H + W) / 2 * 0.003), 2)  # 近似原生 plot 的线宽策略
#             for (box, c) in zip(boxes_f, cls_f):
#                 name = id2name.get(int(c), f'class{int(c)}')
#                 if name == 'vegetation':
#                     color_box = (0, 0, 255)       # 蓝
#                 elif name == 'brick_loss':
#                     color_box = (0, 255, 255)     # 青
#                 else:
#                     color_box = (255, 0, 255)       # 其它类给个默认色
#                 x1, y1, x2, y2 = map(int, box)
#                 cv2.rectangle(cam_image, (x1, y1), (x2, y2), color_box, lw)

#         # 去掉padding边界并保存
#         cam_image = cam_image[top:cam_image.shape[0] - bottom, left:cam_image.shape[1] - right]
#         cam_image = Image.fromarray(cam_image)
#         cam_image.save(save_path)
    
#     def __call__(self, img_path, save_path):
#         # remove dir if exist
#         if os.path.exists(save_path):
#             shutil.rmtree(save_path)
#         # make dir if not exist
#         os.makedirs(save_path, exist_ok=True)

#         if os.path.isdir(img_path):
#             for img_path_ in os.listdir(img_path):
#                 self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}')
#         else:
#             self.process(img_path, f'{save_path}/result.png')
        
# def get_params():
#     params = {
#         'weight':  r'/home/xgq/Desktop/yolo/runs/train/exp69/weights/best.pt', # 现在只需要指定权重即可
#         'device': 'cuda:0',
#         'method': 'GradCAMPlusPlus', # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, KPCA_CAM
#         'layer': [10, 12, 14, 16, 18],
#         'backward_type': 'all', # detect:<class, box, all> ...
#         # === 改动1：两个类别分别设置置信度（0=vegetation, 1=brick_loss）===
#         'conf_threshold': {0: 0.15, 1: 0.15, 2: 0.1},
#         'ratio': 0.02,
#         'show_result': True,  # 画框
#         'renormalize': False, # 若要把热力图限制在框内，设 True
#         'task':'detect',
#         'img_size':640,
#     }
#     return params

# # pip install grad-cam==1.5.4 --no-deps
# if __name__ == '__main__':
#     save_dir = time.strftime("result_%Y%m%d_%H%M%S")
#     model = yolo_heatmap(**get_params())
#     # model(r'C:\path\to\folder', save_dir)
#     # model(r"/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/dataset/6.2/images/val/(1609).JPG", save_dir)
#     model(r'/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/dataset/6.2/images/val/(1).jpg', save_dir)

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch, yaml, cv2, os, shutil, sys, copy, time
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics import YOLO
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, KPCA_CAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (top, bottom, left, right)


# =================================================================
# 自定义 ActivationsAndGradients（保持你原来的逻辑）
# =================================================================
class ActivationsAndGradients:
    """ Class for extracting activations and registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519, we don't use backward hook to record gradients.
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
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'segment':
            logits_ = result[0][:, 4:4 + self.model.nc]
            boxes_ = result[0][:, :4]
            mask_p, mask_nm = result[1][2].squeeze(), result[1][1].squeeze().transpose(1, 0)
            c, h, w = mask_p.size()
            mask = (mask_nm @ mask_p.view(c, -1))
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], mask[indices[0]]
        elif self.model.task == 'pose':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            poses_ = result[:, 4 + self.model.nc:]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(poses_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'obb':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            angles_ = result[:, 4 + self.model.nc:]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(angles_[0], dim0=0, dim1=1)[indices[0]]
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


# =================================================================
# 下面是修过的 target 类 —— 核心修复：返回 tensor，不再返回 int
# =================================================================
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

        # 至少看 1 个框，避免 int(post_result.size(0)*ratio) == 0
        num = max(1, int(post_result.size(0) * self.ratio))

        for i in trange(num):
            if i >= post_result.size(0):
                break

            # 取当前框的置信度
            if self.end2end:
                score = post_result[i, 0]
            else:
                score = post_result[i].max()

            # 低于阈值则停止（后面分数更低）
            if float(score) < self.conf:
                break

            # 分类分数
            if self.ouput_type in ['class', 'all']:
                result.append(score)

            # box 回归
            if self.ouput_type in ['box', 'all']:
                for j in range(4):
                    result.append(pre_post_boxes[i, j])

        if len(result) == 0:
            # 没有任何框通过筛选，用所有框里最大的 score 兜底
            loss = post_result.max()
        else:
            loss = torch.stack(result).sum()

        return loss


class yolo_segment_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)

    def forward(self, data):
        post_result, pre_post_boxes, pre_post_mask = data
        result = []

        num = max(1, int(post_result.size(0) * self.ratio))

        for i in trange(num):
            if i >= post_result.size(0):
                break

            score = post_result[i].max()
            if float(score) < self.conf:
                break

            if self.ouput_type in ['class', 'all']:
                result.append(score)

            if self.ouput_type in ['box', 'all']:
                for j in range(4):
                    result.append(pre_post_boxes[i, j])

            if self.ouput_type in ['segment', 'all']:
                result.append(pre_post_mask[i].mean())

        if len(result) == 0:
            loss = post_result.max()
        else:
            loss = torch.stack(result).sum()

        return loss


class yolo_pose_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)

    def forward(self, data):
        post_result, pre_post_boxes, pre_post_pose = data
        result = []

        num = max(1, int(post_result.size(0) * self.ratio))

        for i in trange(num):
            if i >= post_result.size(0):
                break

            score = post_result[i].max()
            if float(score) < self.conf:
                break

            if self.ouput_type in ['class', 'all']:
                result.append(score)

            if self.ouput_type in ['box', 'all']:
                for j in range(4):
                    result.append(pre_post_boxes[i, j])

            if self.ouput_type in ['pose', 'all']:
                result.append(pre_post_pose[i].mean())

        if len(result) == 0:
            loss = post_result.max()
        else:
            loss = torch.stack(result).sum()

        return loss


class yolo_obb_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)

    def forward(self, data):
        post_result, pre_post_boxes, pre_post_angle = data
        result = []

        num = max(1, int(post_result.size(0) * self.ratio))

        for i in trange(num):
            if i >= post_result.size(0):
                break

            score = post_result[i].max()
            if float(score) < self.conf:
                break

            if self.ouput_type in ['class', 'all']:
                result.append(score)

            if self.ouput_type in ['box', 'all']:
                for j in range(4):
                    result.append(pre_post_boxes[i, j])

            if self.ouput_type in ['obb', 'all']:
                result.append(pre_post_angle[i])

        if len(result) == 0:
            loss = post_result.max()
        else:
            loss = torch.stack(result).sum()

        return loss


class yolo_classify_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)

    def forward(self, data):
        # 分类任务直接最大 logit
        return data.max()


# =================================================================
# 主类：yolo_heatmap
# =================================================================
class yolo_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_result, renormalize, task, img_size):
        device = torch.device(device)
        model_yolo = YOLO(weight)
        model_names = model_yolo.names
        print(f'model class info:{model_names}')
        model = copy.deepcopy(model_yolo.model)
        model.to(device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()

        model.task = task
        if not hasattr(model, 'end2end'):
            model.end2end = False

        # 若传 dict，则用最小阈值供 CAM 反传用
        if isinstance(conf_threshold, dict):
            conf_for_cam = float(min(conf_threshold.values()))
        else:
            conf_for_cam = float(conf_threshold)

        if task == 'detect':
            target = yolo_detect_target(backward_type, conf_for_cam, ratio, model.end2end)
        elif task == 'segment':
            target = yolo_segment_target(backward_type, conf_for_cam, ratio, model.end2end)
        elif task == 'pose':
            target = yolo_pose_target(backward_type, conf_for_cam, ratio, model.end2end)
        elif task == 'obb':
            target = yolo_obb_target(backward_type, conf_for_cam, ratio, model.end2end)
        elif task == 'classify':
            target = yolo_classify_target(backward_type, conf_for_cam, ratio, model.end2end)
        else:
            raise Exception(f"not support task({task}).")

        target_layers = [model.model[l] for l in layer]

        print("Target layers:")
        for tl in target_layers:
            print(tl)

        cam_method = eval(method)(model, target_layers)
        cam_method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

        # 保存到实例
        self.model_yolo = model_yolo
        self.model = model
        self.device = device
        self.method = cam_method
        self.target = target
        self.conf_threshold = conf_threshold
        self.ratio = ratio
        self.show_result = show_result
        self.renormalize = renormalize
        self.task = task
        self.img_size = img_size

        # 自定义框颜色（BGR）
        self.color_map_bgr = {
            'vegetation': (0, 0, 255),
            'brick_loss': (0, 255, 255),
            'deadmood': (255, 0, 255)
        }

    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.05)[0]
        return result

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def process(self, img_path, save_path):
        # 读图
        try:
            img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
        except:
            print(f"Warning... {img_path} read failure.")
            return
        img, _, (top, bottom, left, right) = letterbox(img, new_shape=(self.img_size, self.img_size), auto=True)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        print(f'tensor size:{tensor.size()}')
        print(f'tensor device:{tensor.device}, model device:{next(self.model.parameters()).device}')

        # Grad-CAM
        try:
            grayscale_cam = self.method(tensor, [self.target])
        except Exception as e:
            print(f"Warning... self.method(tensor, [self.target]) failure. Error: {repr(e)}")
            import traceback
            traceback.print_exc()
            return

        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        # 预测（用最小阈值拿候选框）
        if isinstance(self.conf_threshold, dict):
            conf_for_predict = float(min(self.conf_threshold.values()))
        else:
            conf_for_predict = float(self.conf_threshold)
        pred = self.model_yolo.predict(tensor, conf=conf_for_predict, iou=0.05)[0]

        # 若无框，直接输出热力图
        if pred.boxes is None or len(pred.boxes) == 0:
            out = Image.fromarray(cam_image[top:cam_image.shape[0] - bottom, left:cam_image.shape[1] - right])
            out.save(save_path)
            return

        # 读取 numpy
        id2name = self.model_yolo.names if isinstance(self.model_yolo.names, dict) \
            else {i: n for i, n in enumerate(self.model_yolo.names)}
        cls_np = pred.boxes.cls.cpu().numpy().astype(int)
        conf_np = pred.boxes.conf.cpu().numpy()
        boxes = pred.boxes.xyxy.cpu().numpy().astype(int)

        # 按类别阈值二次过滤
        if isinstance(self.conf_threshold, dict):
            keep_mask = np.array(
                [conf_np[i] >= float(self.conf_threshold.get(int(cls_np[i]), conf_for_predict))
                 for i in range(len(cls_np))],
                dtype=bool
            )
        else:
            keep_mask = conf_np >= float(self.conf_threshold)
        boxes_f = boxes[keep_mask]
        cls_f = cls_np[keep_mask]

        # 可选：把 CAM 限定在框内
        if self.renormalize and self.task in ['detect', 'segment', 'pose'] and len(boxes_f) > 0:
            cam_image = self.renormalize_cam_in_bounding_boxes(boxes_f, img, grayscale_cam)

        # 画框
        if self.show_result and len(boxes_f) > 0:
            H, W = cam_image.shape[:2]
            lw = max(round((H + W) / 2 * 0.003), 2)
            for (box, c) in zip(boxes_f, cls_f):
                name = id2name.get(int(c), f'class{int(c)}')
                if name == 'vegetation':
                    color_box = (0, 0, 255)   # 蓝
                elif name == 'brick_loss':
                    color_box = (0, 255, 255)  # 青
                else:
                    color_box = (255, 0, 255)  # 其它类默认紫
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(cam_image, (x1, y1), (x2, y2), color_box, lw)

        # 去 padding & 保存
        cam_image = cam_image[top:cam_image.shape[0] - bottom, left:cam_image.shape[1] - right]
        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)

    def __call__(self, img_path, save_path):
        # remove dir if exist
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            for img_path_ in os.listdir(img_path):
                self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}')
        else:
            self.process(img_path, f'{save_path}/result.png')


def get_params():
    params = {
        'weight': r'/home/xgq/Desktop/yolo/runs/train/exp72/weights/best.pt',  # 现在只需要指定权重即可
        'device': 'cuda:0',
        'method': 'GradCAMPlusPlus',  # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, KPCA_CAM
        'layer': [10, 12, 14, 16, 18],
        'backward_type': 'all',  # detect:<class, box, all> ...
        'conf_threshold': {0: 0.15, 1: 0.15, 2: 0.08},
        'ratio': 0.02,
        'show_result': True,   # 画框
        'renormalize': False,  # 若要把热力图限制在框内，设 True
        'task': 'detect',
        'img_size': 640,
    }
    return params


# pip install grad-cam==1.5.4 --no-deps
if __name__ == '__main__':
    save_dir = time.strftime("result_%Y%m%d_%H%M%S")
    model = yolo_heatmap(**get_params())
    model(r'/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/dataset/6.2/images/val/(1618).jpg', save_dir)
