import os
import csv
from pathlib import Path

import cv2
import numpy as np

# =========================
# 1. 路径配置
# =========================
dataset_a_dir = r"/home/xgq/Desktop/天气光照实验/原始/images"
dataset_b_dir = r"/home/xgq/Desktop/天气光照实验/雨天/images"

output_csv = r"/home/xgq/Desktop/match_results_sift.csv"
vis_dir = r"/home/xgq/Desktop/match_vis_sift1"

# =========================
# 2. 参数配置
# =========================
img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 每张A图只保留B中最好的前k个候选
topk = 3

# 图像最长边缩放到这个值，加快匹配
max_side = 1200

# Lowe ratio test 阈值，越小越严格
ratio_thresh = 0.75

# 最少“好匹配点”数量，低于这个基本不可信
min_good_matches = 18

# RANSAC后最少内点数
min_inliers = 10

# 内点比例阈值：inliers / good_matches
min_inlier_ratio = 0.35

# 是否要求双向唯一匹配（推荐True）
mutual_best_match = True

# =========================
# 3. 创建输出目录
# =========================
os.makedirs(vis_dir, exist_ok=True)

# =========================
# 4. 初始化SIFT和BFMatcher
# =========================
# 若你的OpenCV不支持SIFT，可改成ORB版本（我后面也给你）
sift = cv2.SIFT_create(
    nfeatures=3000,
    contrastThreshold=0.02,
    edgeThreshold=10,
    sigma=1.6
)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


# =========================
# 5. 工具函数
# =========================
def get_image_files(folder):
    files = []
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        if os.path.isfile(p) and Path(f).suffix.lower() in img_exts:
            files.append(p)
    files.sort()
    return files


def resize_keep_ratio(img, max_side=1200):
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def read_image_gray(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    img = resize_keep_ratio(img, max_side=max_side)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def extract_sift_features(img_path):
    try:
        img, gray = read_image_gray(img_path)
        kpts, desc = sift.detectAndCompute(gray, None)
        return {
            "path": img_path,
            "img": img,
            "gray": gray,
            "kpts": kpts,
            "desc": desc
        }
    except Exception as e:
        print(f"[错误] 提取特征失败: {img_path}, {e}")
        return None


def match_features(feat_a, feat_b):
    """
    返回：
    good_matches数量、RANSAC内点数、内点比例、是否通过、可视化需要的数据
    """
    if feat_a is None or feat_b is None:
        return None

    desc_a = feat_a["desc"]
    desc_b = feat_b["desc"]
    kpts_a = feat_a["kpts"]
    kpts_b = feat_b["kpts"]

    if desc_a is None or desc_b is None:
        return {
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "passed": False,
            "matches_for_draw": [],
            "mask": None
        }

    if len(desc_a) < 2 or len(desc_b) < 2:
        return {
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "passed": False,
            "matches_for_draw": [],
            "mask": None
        }

    # knn匹配
    knn_matches = bf.knnMatch(desc_a, desc_b, k=2)

    # Lowe ratio test
    good = []
    for m_n in knn_matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    good_matches = len(good)

    if good_matches < min_good_matches:
        return {
            "good_matches": good_matches,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "passed": False,
            "matches_for_draw": good,
            "mask": None
        }

    pts_a = np.float32([kpts_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_b = np.float32([kpts_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # RANSAC估计单应性矩阵
    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 5.0)

    if mask is None:
        return {
            "good_matches": good_matches,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "passed": False,
            "matches_for_draw": good,
            "mask": None
        }

    mask = mask.ravel().tolist()
    inliers = int(sum(mask))
    inlier_ratio = inliers / max(good_matches, 1)

    passed = (
        good_matches >= min_good_matches and
        inliers >= min_inliers and
        inlier_ratio >= min_inlier_ratio
    )

    return {
        "good_matches": good_matches,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "passed": passed,
        "matches_for_draw": good,
        "mask": mask
    }


def save_match_visualization(feat_a, feat_b, match_info, save_path):
    try:
        img_vis = cv2.drawMatches(
            feat_a["img"], feat_a["kpts"],
            feat_b["img"], feat_b["kpts"],
            match_info["matches_for_draw"], None,
            matchesMask=match_info["mask"] if match_info["mask"] is not None else None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imencode('.jpg', img_vis)[1].tofile(save_path)
    except Exception as e:
        print(f"[错误] 保存可视化失败: {save_path}, {e}")


def score_match(match_info):
    """
    给候选排序用的综合分数
    """
    return (
        match_info["inliers"] * 3.0 +
        match_info["good_matches"] * 1.0 +
        match_info["inlier_ratio"] * 100.0
    )


# =========================
# 6. 读取图片列表
# =========================
a_files = get_image_files(dataset_a_dir)
b_files = get_image_files(dataset_b_dir)

print(f"数据集A图片数量: {len(a_files)}")
print(f"数据集B图片数量: {len(b_files)}")

if len(a_files) == 0 or len(b_files) == 0:
    raise ValueError("有一个数据集文件夹中没有图片，请检查路径。")

# =========================
# 7. 提取全部特征
# =========================
print("正在提取数据集A特征...")
a_feats = {}
for i, path in enumerate(a_files, 1):
    a_feats[path] = extract_sift_features(path)
    print(f"A: {i}/{len(a_files)}", end="\r")
print()

print("正在提取数据集B特征...")
b_feats = {}
for i, path in enumerate(b_files, 1):
    b_feats[path] = extract_sift_features(path)
    print(f"B: {i}/{len(b_files)}", end="\r")
print()

# =========================
# 8. A->B 匹配
# =========================
forward_best = {}   # A中每张图在B中的最佳匹配
all_forward_results = []

print("开始 A -> B 匹配...")
for i, a_path in enumerate(a_files, 1):
    feat_a = a_feats[a_path]
    candidates = []

    for b_path in b_files:
        feat_b = b_feats[b_path]
        match_info = match_features(feat_a, feat_b)
        if match_info is None:
            continue

        candidates.append({
            "a_path": a_path,
            "b_path": b_path,
            "good_matches": match_info["good_matches"],
            "inliers": match_info["inliers"],
            "inlier_ratio": match_info["inlier_ratio"],
            "passed": match_info["passed"],
            "score": score_match(match_info),
            "match_info": match_info
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = candidates[:topk]
    all_forward_results.extend(top_candidates)

    if len(candidates) > 0:
        forward_best[a_path] = candidates[0]

    print(f"A->B: {i}/{len(a_files)}", end="\r")
print()

# =========================
# 9. 若启用，做 B->A 反向匹配，提升准确性
# =========================
backward_best = {}

if mutual_best_match:
    print("开始 B -> A 匹配...")
    for i, b_path in enumerate(b_files, 1):
        feat_b = b_feats[b_path]
        candidates = []

        for a_path in a_files:
            feat_a = a_feats[a_path]
            match_info = match_features(feat_b, feat_a)
            if match_info is None:
                continue

            candidates.append({
                "b_path": b_path,
                "a_path": a_path,
                "good_matches": match_info["good_matches"],
                "inliers": match_info["inliers"],
                "inlier_ratio": match_info["inlier_ratio"],
                "passed": match_info["passed"],
                "score": score_match(match_info)
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        if len(candidates) > 0:
            backward_best[b_path] = candidates[0]

        print(f"B->A: {i}/{len(b_files)}", end="\r")
    print()

# =========================
# 10. 汇总结果
# =========================
final_results = []

print("正在生成最终结果...")
for item in all_forward_results:
    a_path = item["a_path"]
    b_path = item["b_path"]

    mutual_ok = True
    if mutual_best_match:
        if b_path not in backward_best:
            mutual_ok = False
        else:
            best_back_a = backward_best[b_path]["a_path"]
            mutual_ok = (best_back_a == a_path)

    final_pass = item["passed"] and mutual_ok

    final_results.append([
        os.path.basename(a_path),
        os.path.basename(b_path),
        item["good_matches"],
        item["inliers"],
        round(item["inlier_ratio"], 4),
        round(item["score"], 4),
        "YES" if item["passed"] else "NO",
        "YES" if mutual_ok else "NO",
        "YES" if final_pass else "NO"
    ])

    # 只给最终通过且是rank1的结果生成可视化
    if final_pass:
        best_of_a = forward_best.get(a_path, None)
        if best_of_a is not None and best_of_a["b_path"] == b_path:
            save_name = (
                f"{Path(a_path).stem}__{Path(b_path).stem}"
                f"__g{item['good_matches']}_i{item['inliers']}"
                f"_r{item['inlier_ratio']:.2f}.jpg"
            )
            save_path = os.path.join(vis_dir, save_name)
            save_match_visualization(
                a_feats[a_path],
                b_feats[b_path],
                item["match_info"],
                save_path
            )

# =========================
# 11. 保存CSV
# =========================
with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_A",
        "image_B",
        "good_matches",
        "inliers",
        "inlier_ratio",
        "score",
        "passed_threshold",
        "mutual_best",
        "final_match"
    ])
    writer.writerows(final_results)

print(f"\n匹配完成！结果已保存到: {output_csv}")
print(f"可视化结果保存在: {vis_dir}")