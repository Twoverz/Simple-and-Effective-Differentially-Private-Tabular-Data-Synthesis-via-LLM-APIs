import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder

# ======================
# 1. 数据读取
# ======================
def load_data1(real_path, syn_path):
    real = pd.read_csv(real_path)
    syn = pd.read_csv(syn_path)

    X_real = real.iloc[:, :-1].values
    y_real = real.iloc[:, -1].values

    X_syn = syn.iloc[:, :-1].values
    y_syn = syn.iloc[:, -1].values

    return X_real, y_real, X_syn, y_syn

def load_data(real, syn):
    X_real = real.iloc[:, :-1].values
    y_real = real.iloc[:, -1].values

    X_syn = syn.iloc[:, :-1].values
    y_syn = syn.iloc[:, -1].values

    return X_real, y_real, X_syn, y_syn

# ======================
# 2. 预处理
# ======================
def preprocess(X_real, X_syn):
    scaler = StandardScaler()
    X_real = scaler.fit_transform(X_real)
    X_syn = scaler.transform(X_syn)
    return X_real, X_syn


# ======================
# 3. 模型定义
# ======================
class TargetModel(nn.Module):
    def __init__(self, d, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class AttackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze()


# ======================
# 4. 训练函数
# ======================
def train_target(model, X, y, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    for _ in range(epochs):
        logits = model(X)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def train_attack(model, X, y, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for _ in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


# ======================
# 5. 特征提取（MIA核心）
# ======================
def compute_features(model, X, y):
    model.eval()

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=1)

        loss = F.cross_entropy(logits, y, reduction='none')
        max_conf, _ = probs.max(dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

        feat = torch.stack([loss, max_conf, entropy], dim=1)

    return feat


# ======================
# 6. 构造攻击数据
# ======================
def build_attack_data(model, X_train, y_train, X_test, y_test):
    feat_train = compute_features(model, X_train, y_train)
    feat_test = compute_features(model, X_test, y_test)

    X_attack = torch.cat([feat_train, feat_test])
    y_attack = torch.cat([
        torch.ones(len(feat_train)),
        torch.zeros(len(feat_test))
    ])

    return X_attack, y_attack


# ======================
# 7. 评估
# ======================
def evaluate(model, X, y):
    with torch.no_grad():
        pred = model(X).numpy()
        y = y.numpy()

    auc = roc_auc_score(y, pred)
    fpr, tpr, _ = roc_curve(y, pred)

    idx = np.where(fpr <= 0.01)[0]
    tpr_1 = tpr[idx[-1]] if len(idx) > 0 else 0

    return auc, tpr_1


# ======================
# 8. 主流程
# ======================
def run_experiment(real_csv, syn_csv):

    # 读取
    X_real, y_real, X_syn, y_syn = load_data(real_csv, syn_csv)

    # 预处理
    X_real, X_syn = preprocess(X_real, X_syn)

    # split real data（用于攻击）
    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
        X_real, y_real, test_size=0.5, random_state=42
    )

    # 类别数
    num_classes = len(np.unique(y_real))

    # 训练 target model（只用合成数据）
    model = TargetModel(X_syn.shape[1], num_classes)
    model = train_target(model, X_syn, y_syn)

    # 构造攻击数据
    X_attack, y_attack = build_attack_data(
        model,
        X_r_train, y_r_train,
        X_r_test, y_r_test
    )

    # 训练攻击模型
    attack_model = AttackModel()
    attack_model = train_attack(attack_model, X_attack, y_attack)

    # 评估
    auc, tpr1 = evaluate(attack_model, X_attack, y_attack)

    print("===== Attack Result =====")
    print(f"AUC: {auc:.4f}")
    print(f"TPR@1%FPR: {tpr1:.4f}")


def encode_categorical_consistent(real_path, syn_path):

    # 读取数据
    real = pd.read_csv(real_path)
    syn = pd.read_csv(syn_path)

    # 找到类别列（字符串类型）
    cat_cols = real.select_dtypes(include=["object"]).columns.tolist()

    print("Categorical columns:", cat_cols)

    # 为每一列构造共享 encoder
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()

        # 合并两边数据进行fit（关键点）
        combined = pd.concat([real[col], syn[col]], axis=0).astype(str)

        le.fit(combined)

        # transform
        real[col] = le.transform(real[col].astype(str))
        syn[col] = le.transform(syn[col].astype(str))

        encoders[col] = le

    return real, syn


# ======================
# 9. 运行
# ======================
if __name__ == "__main__":

    real, syn = encode_categorical_consistent(
    "F:/csv/synthetic_data_llm_sqe/squ_kdd_private_data_eps10.0.csv",
    "F:/csv/synthetic_data_llm_sqe/squ_kdd_syn_data_eps10.0.csv"
)
#     real, syn = encode_categorical_consistent(
#         "C:/Users/agang/PycharmProjects/llm_pe/Real_data/raw_csv/adult.csv",
#         "F:/csv/Syn_Datasets/DP_CTGAN/eps10/DP_CTGAN_adult.csv"
#     )
    run_experiment(real, syn)
    # run_experiment("synthetic_data/adult_private_data_eps1.0.csv", "synthetic_data/adult_syn_data_eps1.0.csv")