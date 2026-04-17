import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('CWADE-net.yaml')
    model.load('yolo11n.pt') # loading pretrain weights
    model.train(data=r"/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/data.yaml",
                cache=False,
                imgsz=640,
                epochs=2,
                batch=8,
                close_mosaic=0, #              
                workers=8,
                # device='0,1',
                optimizer='SGD', 
                # patience=0, 
                # resume=True, 
                amp=False, 
                # fraction=0.2, 
                project='runs/train',
                name='exp',
                )
