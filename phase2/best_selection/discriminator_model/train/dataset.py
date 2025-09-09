import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 路径配置
ALL_FEATURES_CSV = "" # /features.csv
HUMAN_LABEL_CSV = "" # /labels.csv
OUTPUT_DIR = "" # /

# 加载特征和人工标注
df_feat = pd.read_csv(ALL_FEATURES_CSV)
df_label = pd.read_csv(HUMAN_LABEL_CSV)

df_label['original_video_name'] = df_label['original_video_name'].str.replace('.mp4', '', regex=False)
df_label['best_video_name'] = df_label['best_video_name'].str.replace('.mp4', '', regex=False)

# 提取组名
def extract_group_key(name):
    return "_".join(name.split("_")[:5])

df_feat['group'] = df_feat['video_name'].apply(extract_group_key)
df_label['group'] = df_label['best_video_name'].apply(extract_group_key)

# 添加 label
df_feat['label'] = 0
for _, row in df_label.iterrows():
    group = row['group']
    original = row['original_video_name']
    df_feat.loc[(df_feat['group'] == group) & (df_feat['video_name'] == original), 'label'] = 1

# 构造 pairwise 训练样本
features = ['imaging_quality', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'similarity']

X_pairs, y_pairs, name_a_list, name_b_list = [], [], [], []

for _, group_df in df_feat.groupby('group'):
    good = group_df[group_df['label'] == 1]
    bad = group_df[group_df['label'] == 0]
    for _, row_g in good.iterrows():
        for _, row_b in bad.iterrows():
            feat_g = row_g[features].values.astype(np.float32)
            feat_b = row_b[features].values.astype(np.float32)

            # 正向：g > b
            X_pairs.append(np.concatenate([feat_g, feat_b]))
            y_pairs.append(1)
            name_a_list.append(row_g['video_name'])
            name_b_list.append(row_b['video_name'])

            # 反向：b < g
            X_pairs.append(np.concatenate([feat_b, feat_g]))
            y_pairs.append(0)
            name_a_list.append(row_b['video_name'])
            name_b_list.append(row_g['video_name'])

# 转 numpy
X = np.array(X_pairs)
y = np.array(y_pairs)
name_a = np.array(name_a_list)
name_b = np.array(name_b_list)

print(f"构建完成训练对数：{len(X)}，输入维度：{X.shape}")

# 划分数据集
X_train, X_temp, y_train, y_temp, name_a_train, name_a_temp, name_b_train, name_b_temp = train_test_split(
    X, y, name_a, name_b, test_size=0.25, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test, name_a_val, name_a_test, name_b_val, name_b_test = train_test_split(
    X_temp, y_temp, name_a_temp, name_b_temp, test_size=0.6, random_state=42, shuffle=True
)

# 数据后处理：高斯平滑similarity中的0值
def replace_zero_similarity_in_array(X_array):
    idx_a_sim = 4
    idx_b_sim = 9
    mu = 0.1
    sigma = 0.07
    low, high = 0.0, 0.25

    def sample_in_range():
        val = np.random.normal(mu, sigma)
        while val < low or val > high:
            val = np.random.normal(mu, sigma)
        return val

    zero_indices_a = np.where(X_array[:, idx_a_sim] == 0)[0]
    zero_indices_b = np.where(X_array[:, idx_b_sim] == 0)[0]

    for i in zero_indices_a:
        X_array[i, idx_a_sim] = sample_in_range()
    for i in zero_indices_b:
        X_array[i, idx_b_sim] = sample_in_range()


# 对三部分数据替换
replace_zero_similarity_in_array(X_train)
replace_zero_similarity_in_array(X_val)
replace_zero_similarity_in_array(X_test)

# 保存为 .npz
np.savez(OUTPUT_DIR + "train.npz", X=X_train, y=y_train, name_a=name_a_train, name_b=name_b_train)
np.savez(OUTPUT_DIR + "val.npz", X=X_val, y=y_val, name_a=name_a_val, name_b=name_b_val)
np.savez(OUTPUT_DIR + "test.npz", X=X_test, y=y_test, name_a=name_a_test, name_b=name_b_test)



