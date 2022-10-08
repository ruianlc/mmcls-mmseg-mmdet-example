"""
公共性函数或者类
"""
import math
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
from time import time
from random import seed, sample

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import make_scorer, accuracy_score, r2_score, log_loss, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold, RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import set_config
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from numpy.linalg import matrix_rank as rank
from sympy import group

set_config(display="diagram")
sns.set_style("ticks")
pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 20)
pd.set_option('mode.chained_assignment', None)
plt.rcParams["figure.max_open_warning"] = 100
filterwarnings("ignore")
plt.rc("font", family="MicroSoft YaHei")

"""
1/全局变量设置
"""
# 设置随机种子，目的是为了结果可重复
random_state = 42
seed(random_state)
np.random.seed(random_state)

# 交叉验证
cv_kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
cv_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
cv_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
cv_rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state)

# 模型评价
scoring_acc = make_scorer(accuracy_score, greater_is_better=True, needs_proba=False)
scoring_r2 = make_scorer(r2_score)
scoring_log = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
scoring_aucc = {'AUC': 'roc_auc', 'Accuracy':make_scorer(accuracy_score)}

"""
2/数据探索相关函数
"""
# 相关系数热力图矩阵
def plot_corr(corr, thresh=.0):
    """
    绘制相关系数矩阵热度图
    corr：相关系数矩阵
    thresh：显示的相关系数阈值
    """
    kot = corr[corr>=thresh]
    plt.figure(figsize=(17, 17))
    c_map = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(kot, annot=True, fmt=".2f", cmap=c_map, square=True, center=0.0)
    plt.show()


# 特征数据分布
def plot_distribution(data,feature_names,figsize=(4.5, 4)):
    for name in feature_names:
        plt.figure(figsize=figsize)
        sns.histplot(x=name, data=data)
        plt.show()

def plot_distribution_multi(data,feature_names,figsize=(12,10)):
    # Plots the histogram for each numerical feature in a separate subplot
    df_X = data[feature_names]
    df_X.hist(bins=25, figsize=figsize, layout=(-1, 4), edgecolor="black")
    plt.tight_layout()
    plt.show()


"""
3/建模相关函数
"""
# 分类模型预测准确率
def evaluate_prediction(y, y_hat, score_func, set_name="test"):
    """
    y: 因变量真实值
    y_hat: 因变量预测值
    set_name: 集合名称, 比如"val", "test"
    """
    score = score_func(y, y_hat)
    print(f"\n{set_name} score: {score}\n")
    print("\nconfusion_matrix:\n")
    print(confusion_matrix(y, y_hat))
    print("\nclassification_report:\n")
    print(classification_report(y, y_hat, digits=6))

    return score


# %% 学习曲线
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 10),
    groups=None,
):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
        groups=groups,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the backbone")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the backbone")

    return plt

# %% 验证曲线
def plot_validation_curve(
    estimator,
    title,
    X,
    y,
    param_name,
    param_range,
    scoring,
    score_cv,
    best_param,
    ylim=(0.0, 1.1),
    cv=None,
    groups=None,
    n_jobs=None,
):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, groups=groups, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(ylim)
    plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="darkorange")
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="navy")
    plt.plot(best_param, score_cv, color='red', marker='o', linestyle="", label="selected_param")
    plt.legend(loc="best")
    plt.grid(True)

    return plt

# 交叉验证，超参优化
class SuperEstimator:
    """
    类似于scikit-learn的model或者transformer，但是，增加了超参优化的功能。
    可以是有监督学习，也可以是无监督学习。
    可以是model，也可以是transformer。
    """

    def __init__(self, searchcv, multiple=0):
        """
        searchcv：可以是GridSearchCV或者RandomizedSearchCV或者BayesSearchCV或者
                HalvingGridSearchCV或者HalvingRandomSearchCV。

        multiple：multiple >= 0。基于验证集成绩的均值与标准差，算出一个综合成绩。
                综合成绩 = 均值 - multiple * 标准差。
                如果全部基于均值，那么，multiple=0。
                如果基于2.5%分位值，假设成绩满足正态分布，那么，multiple=1.96。
                multiple也可以改成其他数字，这个数字越大，表示标准差的影响力越大。
        """
        self.estimator = searchcv.estimator
        self.searchcv = searchcv
        self.multiple = multiple

    def _best_mean_std(self, cv_results_):
        """
        基于验证集成绩的均值与标准差，算出一个综合的成绩。
        """
        scores = np.array(cv_results_["mean_test_score"]) - self.multiple * np.array(
            cv_results_["std_test_score"]
        )
        best_idx = np.argmax(scores)
        return best_idx

    def search(self, X, y=None, groups=None, verbose=True, n_jobs=-1):
        """
        超参优化。
        """
        t0 = time()
        self.searchcv.fit(X, y=y, groups=groups)
        t1 = time()
        best_idx = self._best_mean_std(self.searchcv.cv_results_)
        best_params = dict(self.searchcv.cv_results_["params"][best_idx])
        self.estimator.set_params(**best_params)
        scores_cv = cross_val_score(
            self.estimator,
            X,
            y=y,
            groups=groups,
            scoring=self.searchcv.scoring,
            cv=self.searchcv.cv,
            n_jobs=n_jobs,
        )
        mu = np.mean(scores_cv)
        #best_score = np.max(scores_cv)
        if verbose:
            print(f"\nSearching elapses {t1 - t0:.6f} seconds.\n")
            print(f"\nbest_params: {best_params}\n")
            print(f"\ncv scores：{scores_cv}\n")
            sigma = np.std(scores_cv)
            bounds_left = mu - 1.96 * sigma
            bounds_right = mu + 1.96 * sigma
            print(
                f"\ncv scores(95% CI): {mu:.6f} ± {1.96 * sigma:.6f} = "
                f"({bounds_left:.6f}, {bounds_right:.6f})\n"
            )
        self.estimator.fit(X, y)
        return best_params, mu, sigma


"""
3/数据处理相关函数
"""


# train= reduce_mem_usage(pd.read_csv("../input/train.csv"))
def reduce_mem_usage(df, verbose=True):
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


def compute_vif(df, considered_features):
    """
    通过方差膨胀因子计算多重共线性
    :param df: 特征数据集
    :param considered_features: 特征列表
    :return: vif
    """
    X = df[considered_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1

    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable'] != 'intercept']
    return vif


def mix(X, y, class_range, threshold=0.2):
    seed(0)
    X_mix = X.copy()
    y_mix = y.copy()
    for i in class_range:
        X_sub = X[y == i, :]
        for j in range(X_sub.shape[0]):
            a = X_sub[j, :]
            for k in range(j + 1, X_sub.shape[0]):
                b = X_sub[k, :]
                w = 0.5
                c = w * b + (1.0 - w) * a
                r = random.Random()
                if r <= threshold:
                    X_mix = np.r_[X_mix, c.reshape(1, -1)]
                    y_mix = np.r_[y_mix, i]

    return X_mix, y_mix


def rand_resampling_stratify(X, y, class_range, choose_times):
    """
    对分类模型原始样本进行随机重采样，提高样本数据量
    :param X: 特征数据集
    :param y: 标签数据集
    :param class_range: 类别范围
    :param choose_times: 每种类别循环随机取的次数
    :return: X_new, y_new
    """
    seed(random_state)
    X_new = X.copy()
    y_new = y.copy()
    for i in class_range:
        X_sub = X[y == i, :]  # 一个类别所有特征数据
        for _ in range(1, choose_times + 1):  # 多次随机取
            for j in range(X_sub.shape[0] - 3):
                a = X_sub[j, :].reshape(1, -1)  # 一个类别的单个样本
                the_last = X_sub[j + 1:-1, :]
                row_idx = np.arange(the_last.shape[0])
                np.random.shuffle(row_idx)
                choose_sample = the_last[row_idx[0:2], :]
                new_sample_tmp = np.append(a, choose_sample, axis=0)  # 按行拼接
                new_sample = np.average(new_sample_tmp, axis=0)  # 按列求均值
                X_new = np.r_[X_new, new_sample.reshape(1, -1)]
                y_new = np.r_[y_new, i]

    return X_new, y_new

def rand_resampling_stratify_group(X, y, groups_train, choose_times):
    """
    对分类模型原始样本进行随机重采样，提高样本数据量
    :param X: 特征数据集
    :param y: 标签数据集
    :param groups_train: 组号标签
    :param choose_times: 每种类别循环随机取的次数
    :return: X_new, y_new, np.asarray(group_new)
    """
    X_new = X.copy()
    y_new = y.copy()
    group_new = groups_train.tolist().copy()

    groups_list = np.unique(groups_train)

    for gg in groups_list:
        # 单个组进行抽样
        X_sub = X[groups_train == gg, :]  # 一个组号所有特征数据
        # y_sub = y[groups_train == gg]  # 一个组号所有类别数据
        # group_sub = groups_train[groups_train == gg]

        for i in range(0, choose_times):  # 多次随机取
            seed(random_state + i)  # 初始种子一致，保证不同循序次数重采样结果不同
            for j in range(X_sub.shape[0] - 3):
                grp_X = X_sub[j, :].reshape(1, -1)  # 一个类别的单个样本
                the_last = X_sub[j + 1:-1, :]
                row_idx = np.arange(the_last.shape[0])
                random.shuffle(row_idx)
                choose_sample = the_last[row_idx[0:2], :]
                new_sample_tmp = np.append(grp_X, choose_sample, axis=0)  # 按行拼接
                new_sample = np.average(new_sample_tmp, axis=0)  # 按列求均值
                X_new = np.r_[X_new, new_sample.reshape(1, -1)]
                y_new = np.r_[y_new, int(gg[:1])]
                group_new.append(gg)
    return X_new, y_new, np.asarray(group_new)

def grouped_avg(myArray, N=2):
    """
    数组多行平均
    :param myArray: 原始二维数据
    :param N: 按行平均的行数
    :return: result
    """
    result = np.cumsum(myArray, 0)[N - 1::N] / float(N)
    result[1:] = result[1:] - result[:-1]
    return result


def eliminate_colinear(df_train, feature_names, corr_thresh=0.89):
    """
    剔除共线性特征
    :param df_train: 原始数据
    :param feature_names: 选用特征名称
    :param corr_thresh: 共线性剔除阈值
    :return selected_columns: 保留的特征
    """
    df_tmp = df_train[feature_names]
    corr = df_tmp.corr()
    # plot_corr(corr, corr_thresh)

    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr).iloc[i, j] >= corr_thresh:
                if columns[j]:
                    columns[j] = False
    selected_columns = df_tmp.columns[columns]

    # df_tmp_n = df_train[selected_columns]
    # corr_n = df_tmp_n.corr()
    # plot_corr(corr_n, 0.0)

    return selected_columns


"""
4/文件处理相关函数
"""
def get_project_rootpath():
    path = os.path.realpath(os.curdir)
    while (True):
        for subpath in os.listdir(path):
            if '.idea' in subpath or '.vscode' in subpath:  # 根目录下pycharm项目必然存在'.idea'文件，同理vscode项目必然存在'.vscode'文件
                return path
        path = os.path.dirname(path)

# 判断文件夹是否存在，不存在则创建
# 2022.09.14. Robin Am.
def mk_dir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在    返回 True
    # 不存在  返回 False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False