import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from tabpfn import TabPFNClassifier, TabPFNRegressor
import pandas as pd
import math
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from scipy.stats import rankdata
# ===============================
# Tree-based Continual Release
# ===============================
class TreeAggregator:
    """
    Differentially Private Continual Release via
    Binary Indexed Tree (Fenwick Tree) Mechanism.

    Supports vector-valued inputs.

    Parameters
    ----------
    T : int
        Maximum number of time steps.
    dim : int
        Dimension of vector input.
    sigma : float
        Standard deviation of Gaussian noise per tree node.
    device : torch.device
    dtype : torch.dtype
    """

    def __init__(self, T, dim, sigma, device=None, dtype=torch.float32):
        self.T = T
        self.dim = dim
        self.sigma = sigma
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

        # Fenwick tree storage (1-indexed)
        self.tree = {}

    # --------------------------------------------------
    # Internal: add noise once per node
    # --------------------------------------------------
    def _get_or_init_node(self, idx):
        if idx not in self.tree:
            noise = torch.randn(
                self.dim,
                device=self.device,
                dtype=self.dtype
            ) * self.sigma
            self.tree[idx] = noise.clone()
        return self.tree[idx]

    "无噪声版本"
    # def _get_or_init_node(self, idx):
    #     if idx not in self.tree:
    #         # 无噪声版本：直接生成全 0 张量
    #         noise = torch.zeros(
    #             self.dim,
    #             device=self.device,
    #             dtype=self.dtype
    #         )
    #         self.tree[idx] = noise.clone()
    #     return self.tree[idx]

    # --------------------------------------------------
    # Update: add value at time t
    # --------------------------------------------------
    def add(self, t, value):
        """
        Add vector 'value' at time step t (1-indexed).
        """
        if t < 1 or t > self.T:
            raise ValueError("t out of range")

        idx = t
        while idx <= self.T:
            node = self._get_or_init_node(idx)
            node += value
            idx += idx & -idx  # move to parent

    # --------------------------------------------------
    # Query: prefix sum up to time t
    # --------------------------------------------------
    def query(self, t):
        """
        Return noisy prefix sum S_t = sum_{i=1}^t x_i + noise
        """
        if t < 1 or t > self.T:
            raise ValueError("t out of range")

        result = torch.zeros(
            self.dim,
            device=self.device,
            dtype=self.dtype
        )

        idx = t
        while idx > 0:
            if idx in self.tree:
                result += self.tree[idx]
            idx -= idx & -idx  # move to child

        return result

    # --------------------------------------------------
    # Convenience: add and immediately release prefix
    # --------------------------------------------------
    def add_and_query(self, t, value):
        """
        Add value at time t and return prefix sum S_t.
        """
        self.add(t, value)
        return self.query(t)

# ===============================
# Table Similarity
# ===============================

def normalize_table_torch(X):
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True) + 1e-8
    return (X - mean) / std

def cosine_sim_matrix_torch(X, Y):
    Xn = F.normalize(X, dim=1)
    Yn = F.normalize(Y, dim=1)
    return torch.matmul(Xn, Yn.T)


## Mixed exponential-family similarity
def joint_log_similarity(
    X_num, Y_num,
    X_cat, Y_cat,
    cat_smoothing=1e-3
):
    """
    X_num: (N, d_num)
    Y_num: (M, d_num)
    X_cat: (N, d_cat) int-coded
    Y_cat: (M, d_cat)

    return: sim matrix (N, M)
    """

    # ---------- numeric Gaussian log-likelihood ----------
    num_scales = X_num.std(axis=0) + 1e-6
    num_scales = num_scales.reshape(1, 1, -1)

    diff = X_num[:, None, :] - Y_num[None, :, :]
    num_term = - (diff ** 2 / (2 * num_scales**2)).sum(axis=2)

    # ---------- categorical likelihood ----------
    ###$$## P(x=y)=1, P(x!=y)=smoothing
    match = (X_cat[:, None, :] == Y_cat[None, :, :]).float()
    # cat_term = np.log(match + cat_smoothing).sum(axis=2)
    cat_term = torch.log(match + cat_smoothing).sum(dim=2)

    return num_term + cat_term


def tree_sigma_from_eps_delta(T, C, n, epsilon, delta):
    L = math.log(1.0 / delta)

    # convert (epsilon, delta) -> rho
    rho = (math.sqrt(L + epsilon) - math.sqrt(L))**2

    sensitivity = C / n

    sigma = sensitivity * math.sqrt(math.log(T) / (2 * rho))

    return sigma


def tree_aggregator_sigma(T, C, n, epsilon, delta):
    """
    Compute required sigma for (epsilon, delta)-DP
    with sensitivity = C / n
    """
    sensitivity = C / n
    return (sensitivity * math.sqrt(2 * math.log(T) * math.log(1.25 / delta))) / epsilon

# ===============================
# Private Reweighting (Core)
# ===============================


def private_reweighting_gpu(
    candidate_num,
    private_num,
    candidate_cat,
    private_cat,
    T=1000,
    lr=0.0015,
    device="cuda",
    clip_norm=1.0,
    delta=1e-4,
    epsilon=0.1
):
    # numpy → torch
    cand = torch.tensor(candidate_num, dtype=torch.float32, device=device)
    priv = torch.tensor(private_num, dtype=torch.float32, device=device)


    cand_cat = torch.tensor(candidate_cat, device=device)
    priv_cat = torch.tensor(private_cat, device=device)

    # normalize
    cand = normalize_table_torch(cand)
    priv = normalize_table_torch(priv)

    # similarity matrix (N × M)
    # sim = cosine_sim_matrix_torch(cand, priv)
    sim = joint_log_similarity(
        cand, priv,
        cand_cat, priv_cat,
    )

    N = cand.shape[0]
    log_w = torch.zeros(N, device=device)

    sigma = tree_aggregator_sigma(
        T=T,
        C=clip_norm,
        n=len(priv),
        epsilon=epsilon,
        delta=delta
    )
    # print('sigma is{}', {sigma})

    tree = TreeAggregator(T, N, sigma, device=device)


    for t in range(1, T+1):
        # scores: (N, M)
        scores = log_w[:, None] + sim

        # softmax over candidates for each private point
        probs = torch.softmax(scores, dim=0)

        # expected responsibility
        grad = probs.mean(dim=1)

        # L2 clipping
        norm = torch.linalg.norm(grad)
        if norm > clip_norm:
            grad = grad * (clip_norm / norm)

        # continual release
        noisy_grad = tree.add_and_query(t, grad)

        # exponentiated gradient
        log_w = log_w + lr * noisy_grad
        log_w = log_w - log_w.max()

    w = torch.exp(log_w)
    w = w / w.sum()

    return w

# ===============================
# Sampling Synthetic Table Rows
# ===============================

def sample_synthetic_gpu(candidate_data, weights, n_samples):

    idx = torch.multinomial(weights, n_samples, replacement=True)
    return candidate_data[idx]#candidate_data.iloc[idx]#


def clean_categorical(df, cat_cols):
        df = df.copy()
        for c in cat_cols:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .replace("?", "UNKNOWN")
            )
        return df


def fit_categorical_mapping(df, cat_cols):
    """
    mapping is fitted ONLY on candidate data
    """
    mappings = {}
    for c in cat_cols:
        uniq = df[c].unique().tolist()
        mappings[c] = {v: i for i, v in enumerate(uniq)}
    return mappings

def transform_categorical(df, cat_cols, mappings, unknown_id=-1):
    X_cat = np.zeros((len(df), len(cat_cols)), dtype=int)
    for j, c in enumerate(cat_cols):
        mp = mappings[c]
        X_cat[:, j] = df[c].map(lambda x: mp.get(x, unknown_id)).to_numpy()
    return X_cat
# ===============================
# Demo: End-to-End Example
# ===============================
if __name__ == "__main__":
    np.random.seed(42)

    # -------- private table (真实数据) --------
    # 例如：年龄、收入、教育年限
    private_df= pd.read_csv("Real_data/raw_csv/adult.csv")
    candidate_df = pd.read_csv("datasets_llm/adult_100k_good.csv")

    # candidate_df['8'] = np.select(
    #     [
    #         candidate_df['8'] < 0.5,
    #         (candidate_df['8'] >= 0.5) & (candidate_df['8'] <= 1.5),
    #         candidate_df['8'] > 1.5
    #     ],
    #     [
    #         0.0,
    #         1.08654,
    #         2.17308
    #     ]
    # )
    #
    # candidate_df['16'] = np.select(
    #     [
    #         candidate_df['16'] < 0.1,
    #         (candidate_df['16'] >= 0.1) & (candidate_df['16'] <= 2.5),
    #         candidate_df['16'] > 2.5
    #     ],
    #     [
    #         0.0,
    #         1.27411,
    #         2.54822
    #     ]
    # )
    # candidate_df.to_csv("datasets_llm/higgs_godd.csv", index=False)

    # candidate_df.iloc[:, -1] = (candidate_df.iloc[:, -1] >= 0.5).astype(int)
    # candidate_df.to_csv("datasets_llm/output.csv", index=False)

    numerical_cols = private_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = private_df.select_dtypes(exclude=np.number).columns.tolist()

    private_df = clean_categorical(private_df, categorical_cols)
    candidate_df = clean_categorical(candidate_df, categorical_cols)

    # -------- fit mapping on candidate --------
    cat_mappings = fit_categorical_mapping(candidate_df, categorical_cols)

    # -------- sample private (保持与 num 对齐) --------
    private_df_sampled = private_df.sample(n=1000, random_state=42)
    candidate_df = candidate_df.sample(n=50000, random_state=42)

    # -------- transform categorical --------
    private_data_cat = transform_categorical(
        private_df_sampled, categorical_cols, cat_mappings
    )

    candidate_data_cat = transform_categorical(
        candidate_df, categorical_cols, cat_mappings
    )

    private_data_num = private_df_sampled[numerical_cols].to_numpy()#.sample(n=1000, random_state=42)
    candidate_data_num = candidate_df[numerical_cols].to_numpy()

    epsilons = 10.0


    # -------- DP reweighting --------
    weights = private_reweighting_gpu(
        candidate_data_num,
        private_data_num,
        candidate_data_cat,
        private_data_cat,
        T=1000,
        lr=0.0015,
        device="cuda",
        delta=1e-4,
        epsilon=epsilons
    )

    # -------- sample synthetic table -------- ###################
    idx = torch.multinomial(weights, 1000, replacement=True)
    idx = idx.cpu()
    synthetic_data = candidate_df.iloc[idx]  # candidate_data.iloc[idx]#
    # synthetic_data = sample_synthetic_gpu(
    #     torch.tensor(candidate_df.to_numpy(), device="cuda"),
    #     weights,
    #     n_samples=1000
    # ).cpu().numpy()


    private_df_sampled.to_csv(f'synthetic_data/adult_private_data_eps{epsilons}.csv')
    synthetic_data.to_csv(f'synthetic_data/adult_syn_data_eps{epsilons}.csv')

    print("Synthetic data shape:", synthetic_data)


    def to_distribution(arr1, arr2, bins=30):
        # 数值属性：直方图归一化分布
        min_val = min(np.min(arr1), np.min(arr2))
        max_val = max(np.max(arr1), np.max(arr2))
        hist1, _ = np.histogram(arr1, bins=bins, range=(min_val, max_val), density=True)
        hist2, _ = np.histogram(arr2, bins=bins, range=(min_val, max_val), density=True)

        # 避免0导致JS无法计算
        hist1 = hist1 + 1e-9
        hist2 = hist2 + 1e-9
        return hist1 / hist1.sum(), hist2 / hist2.sum()

    for j, col in enumerate(private_df_sampled.columns):
        print(f"{col:15s}")

        # 获取列数据
        private_col = private_df_sampled.iloc[:, j]
        synthetic_col = synthetic_data.iloc[:, j]

        # 判断是否为分类属性
        if private_col.dtype == 'object' or private_col.dtype.name == 'category':
            # 分类属性：条形图 + JS散度
            plt.figure(figsize=(10, 6))

            private_counts = private_col.value_counts(normalize=True)
            synthetic_counts = synthetic_col.value_counts(normalize=True)

            all_categories = private_counts.index.union(synthetic_counts.index)
            private_counts = private_counts.reindex(all_categories, fill_value=1e-9)
            synthetic_counts = synthetic_counts.reindex(all_categories, fill_value=1e-9)

            x = np.arange(len(all_categories))
            width = 0.35

            plt.bar(x - width / 2, private_counts, width, alpha=0.6, label="Private")
            plt.bar(x + width / 2, synthetic_counts, width, alpha=0.6, label="Synthetic")
            plt.xticks(x, all_categories, rotation=45, ha='right')
            plt.xlabel('Categories')
            plt.ylabel('Frequency')
            plt.legend()
            plt.title(f"Feature: {col} (Categorical)")
            plt.tight_layout()
            plt.show()

            # 计算 JS Divergence
            p = private_counts.values
            q = synthetic_counts.values
            jsd = jensenshannon(p, q, base=2)
            print(f"  JS Divergence (JSD): {jsd:.4f}")

        else:
            # 数值属性：直方图 + Wasserstein Distance
            plt.figure(figsize=(10, 6))
            plt.hist(private_col, bins=30, density=True, alpha=0.6, label="Private")
            plt.hist(synthetic_col, bins=30, density=True, alpha=0.6, label="Synthetic")
            plt.legend()
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title(f"Feature: {col} (Numerical)")
            plt.show()

            # 计算 Wasserstein Distance
            wd = wasserstein_distance(private_col, synthetic_col)
            print(f"  Wasserstein Distance (WD): {wd:.4f}")

            # 数值范围
            try:
                p_min, p_max = private_col.min(), private_col.max()
                s_min, s_max = synthetic_col.min(), synthetic_col.max()
                print(f"  Private range: [{p_min:.4f}, {p_max:.4f}]")
                print(f"  Synthetic range: [{s_min:.4f}, {s_max:.4f}]")
            except Exception as e:
                print(f"  Cannot compute min/max: {e}")

        print("-" * 50)

    # for j, col in enumerate(private_df_sampled.columns):
    #     print(f"{col:15s}")
    #
    #     # 获取列数据
    #     private_col = private_df_sampled.iloc[:, j]
    #     synthetic_col = synthetic_data.iloc[:, j]
    #
    #     # 判断是否为分类属性
    #     # 方法1：检查数据类型
    #     if private_col.dtype == 'object' or private_col.dtype.name == 'category':
    #         # 方法2：或者检查唯一值数量（可选）
    #         # if private_col.nunique() < 20:  # 如果唯一值少于20个，视为分类
    #
    #         # 对于分类属性，使用条形图
    #         plt.figure(figsize=(10, 6))
    #
    #         # 计算频率
    #         private_counts = private_col.value_counts(normalize=True)
    #         synthetic_counts = synthetic_col.value_counts(normalize=True)
    #
    #         # 确保两个系列有相同的索引（类别）
    #         all_categories = private_counts.index.union(synthetic_counts.index)
    #         private_counts = private_counts.reindex(all_categories, fill_value=0)
    #         synthetic_counts = synthetic_counts.reindex(all_categories, fill_value=0)
    #
    #         # 设置条形图位置
    #         x = np.arange(len(all_categories))
    #         width = 0.35
    #
    #         plt.bar(x - width / 2, private_counts, width, alpha=0.6, label="Private")
    #         plt.bar(x + width / 2, synthetic_counts, width, alpha=0.6, label="Synthetic")
    #
    #         plt.xticks(x, all_categories, rotation=45, ha='right')
    #         plt.xlabel('Categories')
    #         plt.ylabel('Frequency')
    #         plt.legend()
    #         plt.title(f"Feature: {col} (Categorical)")
    #         plt.tight_layout()
    #         plt.show()
    #
    #     else:
    #         # 对于数值属性，使用直方图
    #         plt.figure(figsize=(10, 6))
    #         plt.hist(private_col, bins=30, density=True, alpha=0.6, label="Private")
    #         plt.hist(synthetic_col, bins=30, density=True, alpha=0.6, label="Synthetic")
    #         plt.legend()
    #         plt.xlabel('Value')
    #         plt.ylabel('Density')
    #         plt.title(f"Feature: {col} (Numerical)")
    #         plt.show()
    #
    #         # 计算最小最大值（仅对数值属性）
    #         try:
    #             p_min, p_max = private_col.min(), private_col.max()
    #             s_min, s_max = synthetic_col.min(), synthetic_col.max()
    #             print(f"  Private range: [{p_min:.4f}, {p_max:.4f}]")
    #             print(f"  Synthetic range: [{s_min:.4f}, {s_max:.4f}]")
    #         except Exception as e:
    #             print(f"  Cannot compute min/max for this column: {e}")
    #
    #     print("-" * 50)

