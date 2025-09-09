import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

# 路径配置
TRAIN_DATA_PATH = "" # /train.npz
VAL_DATA_PATH = "" # /val.npz
CKPT_PATH = "" # /model.pkl

# 加载数据
def load_npz(path):
    data = np.load(path)
    return data['X'], data['y']

X_train, y_train = load_npz(TRAIN_DATA_PATH)
X_val, y_val = load_npz(VAL_DATA_PATH)


# 模型训练 + 验证评估
def evaluate_model(name):
    print(f"\nEvaluating: {name}")
    

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    # 验证指标
    print("Accuracy:       ", accuracy_score(y_val, y_pred))
    print("F1 Score:       ", f1_score(y_val, y_pred))
    print("AUC Score:      ", roc_auc_score(y_val, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]
joblib.dump(model, CKPT_PATH)
print('over')