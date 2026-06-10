import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# ====================== 1. 请你修改这里 ======================
REAL_CSV_PATH = "Real_data/raw_csv/adult.csv"    # 真实数据路径 shopper_private_data_eps1.0.csv
SYN_CSV_PATH = "F:/csv/Syn_Datasets/DP_CTGAN/eps10/DP_CTGAN_adult.csv"  # 合成数据路径
#F:/csv/synthetic_data_llm/shopper_syn_data_eps1.0.csv
def auto_detect_columns(df, unique_threshold=10):
    """
    自动识别 DataFrame 中的 连续属性 和 分类属性
    :param df: 数据集 DataFrame
    :param unique_threshold: 分类属性最大唯一值数量（默认：唯一值≤10 视为分类）
    :return: continuous_cols, categorical_cols
    """
    continuous_cols = []
    categorical_cols = []

    for col in df.columns:
        # 跳过空值太多的列（可选）
        if df[col].isna().mean() > 0.5:
            continue

        # 1. 非数值类型 → 一定是分类属性
        if not np.issubdtype(df[col].dtype, np.number):
            categorical_cols.append(col)
            continue

        # 2. 数值类型，但唯一值很少 → 视为分类（如性别 0/1、学历编码 1-5）
        unique_count = df[col].nunique()
        if unique_count <= unique_threshold:
            categorical_cols.append(col)
        else:
            continuous_cols.append(col)

    return continuous_cols, categorical_cols
# =============================================================

# 加载数据
real = pd.read_csv(REAL_CSV_PATH)
syn = pd.read_csv(SYN_CSV_PATH)
# real = real.iloc[:, 1:]
# syn = syn.iloc[:, 1:]


CONTINUOUS_COLS, CATEGORICAL_COLS = auto_detect_columns(real)
# CONTINUOUS_COLS = ["age", "income", "score"]  # 你的连续特征列
# CATEGORICAL_COLS = ["gender", "education", "occupation"]  # 你的分类特征列
# 连续特征距离阈值（越小越严格，一般 0.1~0.5 之间）
CONTINUOUS_DIST_THRESHOLD = 0.2

# 只保留需要的列
real = real[CATEGORICAL_COLS + CONTINUOUS_COLS].dropna()
syn = syn[CATEGORICAL_COLS + CONTINUOUS_COLS].dropna()

# 对连续特征标准化（消除量纲影响）
scaler = StandardScaler()
real_cont = scaler.fit_transform(real[CONTINUOUS_COLS])
syn_cont = scaler.transform(syn[CONTINUOUS_COLS])

# 分类特征转字符串（方便精确匹配）
real_cat = real[CATEGORICAL_COLS].astype(str).agg('||'.join, axis=1).values
syn_cat = syn[CATEGORICAL_COLS].astype(str).agg('||'.join, axis=1).values

def calculate_duplication_rate(source_data, compare_data, source_cat, compare_cat, cont_threshold):
    """
    计算重复率：同时满足 分类精确匹配 + 连续特征距离 < 阈值
    """
    n_source = len(source_data)
    duplicate_count = 0

    for i in range(n_source):
        # 当前样本的分类键 + 连续向量
        cat_i = source_cat[i]
        cont_i = source_data[i]

        # 先快速过滤：只和分类完全一样的样本比较
        match_cat_indices = np.where(compare_cat == cat_i)[0]
        if len(match_cat_indices) == 0:
            continue

        # 计算连续特征欧式距离
        cont_compare = compare_data[match_cat_indices]
        distances = np.linalg.norm(cont_compare - cont_i, axis=1)

        # 存在距离小于阈值 = 重复
        if np.any(distances < cont_threshold):
            duplicate_count += 1

    duplication_rate = duplicate_count / n_source
    return duplication_rate, duplicate_count


# ====================== 计算两个核心重复率 ======================
print("="*60)
print("🔍 隐私合成数据 - 重复率评估结果")
print("="*60)


# 2. 真实 vs 合成重复率（合成样本复制了多少真实样本）
syn_vs_real_rate, syn_vs_real_dup = calculate_duplication_rate(
    syn_cont, real_cont, syn_cat, real_cat, CONTINUOUS_DIST_THRESHOLD
)
print(f"✅ 真实-合成跨集重复率: {syn_vs_real_rate:.4f} ({syn_vs_real_dup}/{len(syn)} 条重复)")

print("="*60)
print("📌 指标说明:")
print("- 真实-合成重复率：合成数据抄袭真实数据的比例，越低隐私性越好")
print("="*60)