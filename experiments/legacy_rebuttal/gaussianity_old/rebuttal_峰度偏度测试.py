# In[]
import torch
import numpy as np
from sklearn.covariance import LedoitWolf
from scipy import stats
from sklearn.decomposition import PCA
import torch.nn.functional as F

def select_class_wise_features_across_tasks(features_dir, feature_files, class_ids):
    """
    Select features from multiple classes.
    """
    selected_features = {}
    for f in feature_files:
        print(f)
        task_id = int(f.split('_')[1])
        selected_features[task_id] = {}
        data = torch.load(os.path.join(features_dir, f), map_location='cpu')
        train_features = F.normalize(data['train_features'], dim=-1).numpy()
        train_labels = data['train_labels'].numpy()
        
        for cls in class_ids:
            selected_features[task_id][cls] = train_features[train_labels == cls]
        
    return selected_features

def regularized_mardia(X):
    n, d = X.shape
    mean_vec = np.mean(X, axis=0)
    X_centered = X - mean_vec
    
    # 手动计算协方差矩阵
    cov_matrix = (X_centered.T @ X_centered) / (n - 1)
    
    # 正则化：协方差矩阵 + 0.1 * 单位矩阵
    reg_cov = 0.7 * cov_matrix + 0.3 * np.eye(d)
    
    # 计算精度矩阵（协方差矩阵的逆）
    precision_matrix = np.linalg.inv(reg_cov)
    
    D = X_centered @ precision_matrix @ X_centered.T
    skewness = np.sum(D ** 3) / (n ** 2)
    kurtosis = np.sum(np.diag(D) ** 2) / n
    
    return skewness, kurtosis


def pca_mardia_test(X, n_components=10):
    n, d = X.shape
    if n_components >= n:
        raise ValueError("降维后的维度必须严格小于样本量 n")

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    X_standardized = X_pca / np.std(X_pca, axis=0)
    
    D = X_standardized @ X_standardized.T

    skewness = np.sum(D ** 3) / (n ** 2)
    kurtosis = np.sum(np.diag(D) ** 2) / n
    return skewness, kurtosis

# In[]

# features_dir = "RGDA_WD_2025-12-19-within/DS_imagenet-r/VB16/I20_C20/r4_Full_ht0.05_hk400_kd1.0_TF_DTide_UTTru_C_AS2K_I/Oada_LR5e-06_B16_IT1500/seed_1993/cached_features"
# features_dir = "RGDA_WD_2025-12-19-within/DS_cifar100_224/VB16/I10_C10/r4_Basic_ht0.05_hk400_C_AS2K_I/Oada_LR0.0001_B16_IT1000/seed_1993/cached_features"
# eatures_dir = "RGDA_WD_2025-12-19-within/DS_cars196_224/VB16/I20_C20/r4_Basic_ht0.05_hk400_C_AS2K_I/Oada_LR0.0001_B16_IT1000/seed_1993/cached_features"
features_dir = "RGDA_WD_2025-12-19-within/DS_cars196_224/VB16/I20_C20/r4_Full_ht0.05_hk400_C_AS2K_I/Oada_LR5e-06_B16_IT1000/seed_1993/cached_features"

# features_dir = "RGDA_WD_2025-12-19-within/DS_cars196_224/VB16/I20_C20/r4_Full_ht0.05_hk400_kd1.0_TF_DTide_UTTru_C_AS2K_I/Oada_LR5e-06_B16_IT1500/seed_1993/cached_features"

feature_files = [f for f in os.listdir(features_dir) if f.endswith('_features.pt')]
class_ids = list(range(20))
class_wise_features = select_class_wise_features_across_tasks(features_dir, feature_files, class_ids)

skewness_list = []
kurtosis_list = []

for task_id in range(0, 10):

    task_skewness_list = []
    task_kurtosis_list = []
    for cls in class_ids:
        # skewness, kurtosis = pca_mardia_test(class_wise_features[task_id][cls])
        skewness, kurtosis = regularized_mardia(class_wise_features[task_id][cls])
        task_skewness_list.append(skewness)
        task_kurtosis_list.append(kurtosis)


    skewness_list.append(task_skewness_list)
    kurtosis_list.append(task_kurtosis_list)

skewness_list = np.array(skewness_list)
kurtosis_list = np.array(kurtosis_list)
# %%
