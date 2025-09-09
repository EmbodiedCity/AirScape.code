import os
import argparse
import csv
from sentence_transformers import SentenceTransformer, util
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import Counter

def load_prompt(prompt_csv_path):
    """从 prompt.csv 中读取 {五位编号: prompt}"""
    prompt_dict = {}
    with open(prompt_csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_name = os.path.splitext(row['video_name'])[0]
            parts = video_name.split('_')
            if len(parts) >= 3:
                code = parts[2]
                prompt_dict[code] = row['prompt']
    return prompt_dict


def parse_image_txt(image_txt_path):
    """从 image.txt 中读取相机轨迹 pose 列表 (qw, qx, qy, qz, tx, ty, tz)"""
    poses = []
    with open(image_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                poses.append([qw, qx, qy, qz, tx, ty, tz])
    return poses


def infer_action_description(poses):
    """
    Generates a concise trajectory action description, strictly classifying:
    - Drone translation (based on 3D coordinates change)
    - Drone yaw rotation (based on camera yaw change)
    - Camera gimbal pitch (based on camera pitch change)
    """
    if len(poses) < 2:
        return "The drone stays still."

    poses_np = np.array(poses)
    
    adjusted_quats = poses_np[:, [1, 2, 3, 0]] 
    rotations = R.from_quat(adjusted_quats)
    translations = poses_np[:, 4:]

    all_actions = []

    # --- Drone Translational Analysis ---
    if len(translations) > 1:
        diff_translations = translations[1:] - translations[:-1]
        translation_magnitudes = np.linalg.norm(diff_translations, axis=1)
        
        for i, vec in enumerate(diff_translations):
            if translation_magnitudes[i] < 0.01:
                continue

            normalized_vec = vec / (translation_magnitudes[i] + 1e-8)
            x, y, z = normalized_vec

            abs_x, abs_y, abs_z = abs(x), abs(y), abs(z)
            if abs_z > max(abs_x, abs_y) and abs_z > 0.1:
                all_actions.append("moving forward" if z < 0 else "moving backward")
            elif abs_x > max(abs_y, abs_z) and abs_x > 0.1:
                all_actions.append("moving rightward" if x > 0 else "moving leftward")
            elif abs_y > max(abs_x, abs_z) and abs_y > 0.1:
                all_actions.append("moving upward" if y > 0 else "moving downward")
    
    # --- Rotational Analysis ---
    if len(rotations) > 1:
        rot_threshold = 0.5
        
        for i in range(len(rotations) - 1):
            relative_rotation = rotations[i].inv() * rotations[i+1]
            euler_angles_deg = np.degrees(relative_rotation.as_euler('xyz', degrees=False))
            roll_change, pitch_change, yaw_change = euler_angles_deg

            # rotating
            if abs(yaw_change) > rot_threshold:
                all_actions.append("rotating left" if yaw_change > 0 else "rotating right")
            
            # gimbal adjust
            if abs(pitch_change) > rot_threshold:
                all_actions.append("tilting camera down" if pitch_change > 0 else "tilting camera up")
                
    # --- Summarize Actions ---
    if not all_actions:
        return "The drone stays still."
    
    action_counts = Counter(all_actions)
    sorted_actions = sorted(action_counts.items(), key=lambda item: (-item[1], item[0]))
    
    description_parts = []
    
    unique_trans_phrases = list(dict.fromkeys([action for action, _ in sorted_actions if "moving" in action]))
    if unique_trans_phrases:
        description_parts.append("The drone is " + " and ".join(unique_trans_phrases))

    unique_yaw_phrases = list(dict.fromkeys([action for action, _ in sorted_actions if "yawing" in action]))
    if unique_yaw_phrases:
        if description_parts: 
            description_parts.append(" and " + " and ".join(unique_yaw_phrases))
        else:
            description_parts.append("The drone is " + " and ".join(unique_yaw_phrases))

    unique_gimbal_phrases = list(dict.fromkeys([action for action, _ in sorted_actions if "tilting camera" in action]))
    if unique_gimbal_phrases:
        if description_parts: 
            description_parts.append(" and " + " and ".join(unique_gimbal_phrases))
        else:
            description_parts.append("The camera is " + " and ".join(unique_gimbal_phrases))

    if not description_parts and np.sum(translation_magnitudes) > 0.05:
        return "The drone has subtle movements."
    
    final_description = ", ".join(description_parts) + "."
    final_description = final_description.replace("The drone is and", "The drone is").replace("The camera is and", "The camera is")
    
    return final_description


def collect_outlier_names(outlier_dir):
    """收集所有 outlier 视频名（不含扩展名）"""
    outliers = set()
    if not os.path.exists(outlier_dir):
        return outliers
    for group in os.listdir(outlier_dir):
        gpath = os.path.join(outlier_dir, group)
        if os.path.isdir(gpath):
            for fname in os.listdir(gpath):
                outliers.add(os.path.splitext(fname)[0])
    return outliers


def load_existing_scores(output_csv_path):
    """已存在的评分项，用于断点续传"""
    if not os.path.exists(output_csv_path):
        return set()
    with open(output_csv_path, newline='') as f:
        return set(row[0] for row in csv.reader(f) if row and not row[0].startswith('video_name'))


def main(args):
    model = SentenceTransformer("/data0/trz/jmy/video-rating/all-MiniLM-L6-v2/")
    prompt_dict = load_prompt(args.prompt_csv)
    outlier_names = collect_outlier_names(args.outlier_dir)
    existing = load_existing_scores(args.output_csv)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    csvfile = open(args.output_csv, 'a', newline='')
    writer = csv.writer(csvfile)
    if os.stat(args.output_csv).st_size == 0:
        writer.writerow(['video_name', 'similarity'])

    for fname in os.listdir(args.video_dir):
        if not fname.endswith(".mp4"):
            continue
        name = os.path.splitext(fname)[0]
        if name in existing:
            continue

        # 取视频编号
        parts = name.split('_')
        if len(parts) < 4:
            writer.writerow([name, 0.0])
            continue
        code = parts[2]

        if name in outlier_names:
            writer.writerow([name, 0.0])
            continue

        image_txt = os.path.join(args.image_dir, name, 'sparse', 'images.txt')
        if not os.path.exists(image_txt):
            writer.writerow([name, 0.0])
            continue

        try:
            poses = parse_image_txt(image_txt)
            traj_desc = infer_action_description(poses)
            prompt = prompt_dict.get(code, "")
            if not prompt:
                writer.writerow([name, 0.0])
                continue

            emb1 = model.encode(prompt, convert_to_tensor=True)
            emb2 = model.encode(traj_desc, convert_to_tensor=True)
            score = util.cos_sim(emb1, emb2).item()
            writer.writerow([name, score])
        except Exception as e:
            print(f"Error processing {name}: {e}")
            writer.writerow([name, 0.0])

    csvfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--outlier_dir', type=str, required=True)
    parser.add_argument('--prompt_csv', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    args = parser.parse_args()
    main(args)
