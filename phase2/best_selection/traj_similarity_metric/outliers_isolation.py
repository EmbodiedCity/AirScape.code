import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


def parse_image_txt(sparse_path):
    image_txt_path = os.path.join(sparse_path, "images.txt")
    poses = []
    image_names = []
    with open(image_txt_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") or line == "":
            i += 1
            continue
        parts = line.split()
        if len(parts) < 10:
            i += 2  # skip the 2D point line too
            continue
        # 提取姿态参数
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        image_name = parts[-1]
        poses.append([qw, qx, qy, qz, tx, ty, tz])
        image_names.append(image_name)
        i += 2  # 跳过下一行 POINT2D 数据
    return poses, image_names



def compute_J_matrix(pose_matrix):
    return np.diff(pose_matrix, axis=0)


def detect_outliers(J_matrices, contamination=0.15):
    X = [J.flatten() for J in J_matrices]
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(X)
    outlier_indices = [i for i, p in enumerate(preds) if p == -1]
    return outlier_indices


def visualize_tsne(J_matrices, outlier_indices, video_paths, group_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 每个 J 是 2D，我们把它展平成一维
    X = np.array([J.flatten() for J in J_matrices])

    # 先 PCA 降维
    pca_components = min(10, X.shape[0], X.shape[1])
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X)

    # 再 t-SNE 到 2D
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)

    # 绘图
    plt.figure(figsize=(8, 6))
    for i, (x, y) in enumerate(X_tsne):
        if i in outlier_indices:
            plt.scatter(x, y, c='red', label='Outlier' if 'Outlier' not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.text(x, y, os.path.basename(video_paths[i]), fontsize=6, color='red')
        else:
            plt.scatter(x, y, c='blue', label='Normal' if 'Normal' not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.text(x, y, os.path.basename(video_paths[i]), fontsize=6, color='blue')

    plt.legend()
    plt.title(f"t-SNE Visualization - {group_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{group_name}_tsne.png"), dpi=300)
    plt.close()


def main(args):
    all_videos = [d for d in os.listdir(args.image_dir) if os.path.isdir(os.path.join(args.image_dir, d))]
    groups = {}
    for v in all_videos:
        prefix = "_".join(v.split("_")[:5])
        groups.setdefault(prefix, []).append(v)

    tsne_dir = os.path.join(args.output_dir, "tsne_plots")
    outlier_dir = os.path.join(args.output_dir)

    for group_name, video_list in groups.items():
        print(f"\n>>> 处理分组: {group_name} ({len(video_list)}个视频)")
        J_matrices = []
        video_paths = []

        for video_name in video_list:
            sparse_path = os.path.join(args.image_dir, video_name, "sparse")
            if not os.path.exists(sparse_path):
                print(f"跳过: {sparse_path} 不存在")
                continue
            poses, _ = parse_image_txt(sparse_path)
            if len(poses) < 2:
                print(f"跳过: {video_name} 帧数太少")
                continue
            J = compute_J_matrix(poses)
            J_matrices.append(J)
            video_paths.append(video_name)

        if len(J_matrices) < 3:
            print("数据太少，跳过该组")
            continue

        outlier_indices = detect_outliers(J_matrices)
        if outlier_indices:
            print(f"检测到 {len(outlier_indices)} 个异常视频：")
            for i in outlier_indices:
                src = os.path.join(args.image_dir, video_paths[i])
                dst = os.path.join(outlier_dir, group_name, video_paths[i])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.system(f"cp -r '{src}' '{dst}'")
                print(f" - 移动: {video_paths[i]} -> {dst}")
        else:
            print("未检测到明显异常。")

        # visualize_tsne(J_matrices, outlier_indices, video_paths, group_name, tsne_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="包含视频子文件夹的主目录")
    parser.add_argument("--output_dir", type=str, required=True, help="结果输出目录")
    args = parser.parse_args()
    main(args)
