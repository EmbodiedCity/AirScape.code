import os
import glob
import pandas as pd
import numpy as np
import joblib
import csv

# 路径配置
MODEL_PATH = ""
ALL_FEATURES_CSV = ""
SRC_VIDEO_DIR = ""
OUTPUT_CSV_PATH = ""

# 读取模型和特征
model = joblib.load(MODEL_PATH)
df_feat = pd.read_csv(ALL_FEATURES_CSV)
df_feat.set_index('video_name', inplace=True)

FEATURE_COLUMNS = ['imaging_quality', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'similarity']

# 工具函数
def extract_group_key(video_name): # TODO: modify this function according to custom needs
    parts = video_name.split('_')
    if len(parts) < 6:
        return video_name
    return '_'.join(parts[:5])

def get_video_feature(video_name):
    try:
        row = df_feat.loc[video_name]
        return row[FEATURE_COLUMNS].values.astype(np.float32)
    except KeyError:
        raise ValueError(f"特征文件中找不到视频：{video_name}")

def get_pair_feature(video_a, video_b):
    feat_a = get_video_feature(video_a)
    feat_b = get_video_feature(video_b)
    pair_feat = np.concatenate([feat_a, feat_b]).reshape(1, -1)
    return pair_feat

def selection(video_list):
    competitors = video_list.copy()
    round_idx = 1
    while len(competitors) > 1:
        next_round = []
        for i in range(0, len(competitors), 2):
            if i + 1 == len(competitors):
                next_round.append(competitors[i])
                continue
            a, b = competitors[i], competitors[i+1]
            pair_feat = get_pair_feature(a, b)
            pred = model.predict(pair_feat)[0]
            winner = a if pred == 1 else b
            next_round.append(winner)
        competitors = next_round
        round_idx += 1
    return competitors[0]

# 主流程
def main(SRC_VIDEO_DIR, OUTPUT_CSV_PATH):

    video_paths = glob.glob(os.path.join(SRC_VIDEO_DIR, "*.mp4"))
    video_names = [os.path.basename(p).replace('.mp4', '') for p in video_paths]

    groups = {}
    for vn in video_names:
        key = extract_group_key(vn)
        groups.setdefault(key, []).append(vn)

    results = []
    for group_key, vids in groups.items():
        print(f"\n处理组：{group_key}，视频数：{len(vids)}")
        try:
            best_video = selection(vids)
        except Exception as e:
            print(f"组 {group_key} 筛选失败，原因: {e}")
            best_video = None
        results.append((group_key, best_video))

    with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['group', 'best_video'])
        for group_key, best_vid in results:
            writer.writerow([group_key, best_vid])
    print(f"\n所有组最优视频保存至：{OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main(SRC_VIDEO_DIR, OUTPUT_CSV_PATH)
