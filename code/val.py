# import warnings
# warnings.filterwarnings('ignore')
# import os
# import numpy as np
# from prettytable import PrettyTable
# from ultralytics import YOLO
# from ultralytics.utils.torch_utils import model_info




# def get_weight_size(path):
#     stats = os.stat(path)
#     return f'{stats.st_size / 1024 / 1024:.1f}'

# if __name__ == '__main__':
#     # model_path = '/mnt/sda2/yolo/ultralytics-8.3.39/runs/train/exp8/weights/best.pt'
#     model_path = r'/home/xgq/Desktop/yolo/runs/train/exp9/weights/best.pt'
#     model = YOLO(model_path) 
#     result = model.val(data=r"/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/data.yaml",
#                         split='val', 
#                         imgsz=640,
#                         batch=32,
#                         # iou=0.7,
#                         # rect=False,
#                         # save_json=True, # if you need to cal coco metrice
#                         project='runs/test',
#                         name='exp',
#                         )
    
#     if model.task == 'detect': # 仅目标检测任务适用
#         model_names = list(result.names.values())
#         preprocess_time_per_image = result.speed['preprocess']
#         inference_time_per_image = result.speed['inference']
#         postprocess_time_per_image = result.speed['postprocess']
#         all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image
        
#         n_l, n_p, n_g, flops = model_info(model.model)
        
#         print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
#         print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
#         print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
#         print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
#         print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)

#         model_info_table = PrettyTable()
#         model_info_table.title = "Model Info"
#         model_info_table.field_names = ["GFLOPs", "Parameters", "前处理时间/一张图", "推理时间/一张图", "后处理时间/一张图", "FPS(前处理+模型推理+后处理)", "FPS(推理)", "Model File Size"]
#         model_info_table.add_row([f'{flops:.1f}', f'{n_p:,}', 
#                                   f'{preprocess_time_per_image / 1000:.6f}s', f'{inference_time_per_image / 1000:.6f}s', 
#                                   f'{postprocess_time_per_image / 1000:.6f}s', f'{1000 / all_time_per_image:.2f}', 
#                                   f'{1000 / inference_time_per_image:.2f}', f'{get_weight_size(model_path)}MB'])
#         print(model_info_table)

#         model_metrice_table = PrettyTable()
#         model_metrice_table.title = "Model Metrice"
#         model_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
#         for idx, cls_name in enumerate(model_names):
#             model_metrice_table.add_row([
#                                         cls_name, 
#                                         f"{result.box.p[idx]:.4f}", 
#                                         f"{result.box.r[idx]:.4f}", 
#                                         f"{result.box.f1[idx]:.4f}", 
#                                         f"{result.box.ap50[idx]:.4f}", 
#                                         f"{result.box.all_ap[idx, 5]:.4f}", # 50 55 60 65 70 75 80 85 90 95 
#                                         f"{result.box.ap[idx]:.4f}"
#                                     ])
#         model_metrice_table.add_row([
#                                     "all(平均数据)", 
#                                     f"{result.results_dict['metrics/precision(B)']:.4f}", 
#                                     f"{result.results_dict['metrics/recall(B)']:.4f}", 
#                                     f"{np.mean(result.box.f1):.4f}", 
#                                     f"{result.results_dict['metrics/mAP50(B)']:.4f}", 
#                                     f"{np.mean(result.box.all_ap[:, 5]):.4f}", # 50 55 60 65 70 75 80 85 90 95 
#                                     f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
#                                 ])
#         print(model_metrice_table)

#         with open(result.save_dir / 'paper_data.txt', 'w+') as f:
#             f.write(str(model_info_table))
#             f.write('\n')
#             f.write(str(model_metrice_table))
        
#         print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
#         print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
#         print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
#         print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
#         print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info


def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'


if __name__ == '__main__':
    # ==============================
    # 1. 模型路径
    # ==============================
    model_path = r'/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/runs/train/CWADE-Net/exp394/weights/best.pt'

    # ==============================
    # 2. 数据集配置文件
    # 你的 data.yaml 应该只有一类：
    # names:
    #   0: crack
    # ==============================
    data_yaml = r"/home/xgq/Desktop/yolo/YOLO/ultralytics-yolo11-20250502/ultralytics-yolo11-main/data330.yaml"

    model = YOLO(model_path)

    result = model.val(
        data=data_yaml,
        split='val',
        imgsz=640,
        batch=32,
        project='runs/test',
        name='exp',
    )

    if model.task == 'detect':   # 仅适用于目标检测任务
        # 类别名
        model_names = list(result.names.values())

        # ==============================
        # 3. 速度信息（单位：ms/图）
        # ==============================
        preprocess_time_per_image = result.speed['preprocess']
        inference_time_per_image = result.speed['inference']
        postprocess_time_per_image = result.speed['postprocess']
        all_time_per_image = (
            preprocess_time_per_image +
            inference_time_per_image +
            postprocess_time_per_image
        )

        # ==============================
        # 4. 模型信息
        # ==============================
        n_l, n_p, n_g, flops = model_info(model.model)

        print('-' * 20 + ' 论文上的数据以以下结果为准 ' + '-' * 20)
        print('-' * 20 + ' 论文上的数据以以下结果为准 ' + '-' * 20)
        print('-' * 20 + ' 论文上的数据以以下结果为准 ' + '-' * 20)

        # ==============================
        # 5. 模型信息表
        # ==============================
        model_info_table = PrettyTable()
        model_info_table.title = "Model Info"
        model_info_table.field_names = [
            "GFLOPs",
            "Parameters",
            "前处理时间/张",
            "推理时间/张",
            "后处理时间/张",
            "Latency/张(总)",
            "FPS(总)",
            "FPS(纯推理)",
            "Model File Size"
        ]

        model_info_table.add_row([
            f'{flops:.1f}',
            f'{n_p:,}',
            f'{preprocess_time_per_image / 1000:.6f}s',
            f'{inference_time_per_image / 1000:.6f}s',
            f'{postprocess_time_per_image / 1000:.6f}s',
            f'{all_time_per_image / 1000:.6f}s',
            f'{1000 / all_time_per_image:.2f}',
            f'{1000 / inference_time_per_image:.2f}',
            f'{get_weight_size(model_path)}MB'
        ])
        print(model_info_table)

        # ==============================
        # 6. 检测指标表
        # ==============================
        model_metrics_table = PrettyTable()
        model_metrics_table.title = "Model Metrics"
        model_metrics_table.field_names = [
            "Class Name",
            "Precision",
            "Recall",
            "F1-Score",
            "mAP50",
            "mAP75",
            "mAP50-95"
        ]

        for idx, cls_name in enumerate(model_names):
            # 单类别情况下通常 idx=0，对应 crack
            precision = result.box.p[idx]
            recall = result.box.r[idx]
            f1_score = result.box.f1[idx]
            map50 = result.box.ap50[idx]
            map75 = result.box.all_ap[idx, 5]   # all_ap: 0.50,0.55,...,0.95，所以索引5对应0.75
            map5095 = result.box.ap[idx]

            model_metrics_table.add_row([
                cls_name,
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1_score:.4f}",
                f"{map50:.4f}",
                f"{map75:.4f}",
                f"{map5095:.4f}"
            ])

        # all 平均行（单类别时，这一行和 crack 的值通常一样）
        model_metrics_table.add_row([
            "all(平均数据)",
            f"{result.results_dict['metrics/precision(B)']:.4f}",
            f"{result.results_dict['metrics/recall(B)']:.4f}",
            f"{np.mean(result.box.f1):.4f}",
            f"{result.results_dict['metrics/mAP50(B)']:.4f}",
            f"{np.mean(result.box.all_ap[:, 5]):.4f}",
            f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
        ])

        print(model_metrics_table)

        # ==============================
        # 7. 保存到 txt
        # ==============================
        save_txt_path = result.save_dir / 'paper_data.txt'
        with open(save_txt_path, 'w+', encoding='utf-8') as f:
            f.write(str(model_info_table))
            f.write('\n\n')
            f.write(str(model_metrics_table))

        print('-' * 20, f'结果已保存至 {save_txt_path}', '-' * 20)
        print('-' * 20, f'结果已保存至 {save_txt_path}', '-' * 20)
        print('-' * 20, f'结果已保存至 {save_txt_path}', '-' * 20)