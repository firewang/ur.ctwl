# -*- encoding: utf-8 -*-
# @Version : 1.0  
# @Time    : 2019/7/29 10:01
# @Author  : wanghd
# @note    : 分类任务全流程搭建，eda 部分抽离作图函数，特征工程部分抽离特征处理函数（类），
# @note    : 模型调参部分抽离学习器，评估方法，参数字典

import os
import numpy as np
import pandas as pd
from math import ceil, sqrt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report as cr
import matplotlib.pyplot as plt
import seaborn as sns
from figure_utils import df_box_plot, df_barplot, df_pair_boxplot, df_pair_plot
from feature_engineer_utils import CustomDummifier, CustomEncoder, my_pipeline

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# sns.set(font="SimHei")


def get_data(filename, filepath=os.getcwd()):
    """读取初始数据"""
    full_path = os.path.join(filepath, filename)
    if os.path.exists(full_path):
        suffix = filename.split(".")[-1]
        if suffix == 'csv':
            df = pd.read_csv(full_path)
        else:
            df = pd.read_excel(full_path)
        return df
    else:
        raise FileNotFoundError


def split_data(df, label_col_name, test_size=0.2):
    label = df.pop(label_col_name)
    train_x, test_x, train_y, test_y = train_test_split(df, label, random_state=11, stratify=label, test_size=test_size)
    return train_x, test_x, train_y, test_y


def eda_data(df):
    """初步的数据探索"""
    # 先查看数据基本情况
    df = df.replace("unknown", np.nan)

    # 缺失比例，均值，标准差，分位数等
    df_info = pd.DataFrame(df.isnull().sum(axis=0), columns=["nan_nums"])
    df_info.loc[:, 'nan_percent'] = df_info.loc[:, 'nan_nums'] / df.shape[0]
    df_info.loc[:, 'nunique'] = df.apply(pd.Series.nunique)  # 各列不同值的数量
    df_info = pd.merge(df_info, df.describe().T, left_index=True, right_index=True, how='left')
    print(df_info)

    # 初步数据探索
    # 数值型单变量分布
    df_box_plot(df.loc[:, ['age', 'balance', 'day']], sub_title='')
    # 离散型单变量分布
    df_barplot(df, sub_title='')
    # 离散型+ 数值型 单变量分布
    df_pair_boxplot(df, 'housing', ['age', ], 'y', figsize=(12, 16))
    # 数值型双变量相关性
    df_pair_plot(df.loc[:, ['age', 'balance', 'day', 'job', 'y']])
    return None


def basic_data_treatment(df):
    """初步数据处理（预处理）"""
    # 类别型热编码
    dummy = CustomDummifier(cols=['marital', 'contact', 'job', "poutcome", 'month'])
    # df = dummy.fit_transform(df)
    # 序列型数值编码
    label_encoder = CustomEncoder(col='education', ordering=['unknown', "primary", "secondary", 'tertiary'])
    # df = label_encoder.fit_transform(df)

    for col in ["default", "housing", 'loan']:
        df.loc[:, col] = df.loc[:, col].map({"no": 0, "yes": 1})

    uf_pipeline = my_pipeline(["dum", 'encoder'], [dummy, label_encoder])
    df = uf_pipeline.fit_transform(df)
    # pd.set_option('display.max_columns', 30)
    # print(df)
    return df


def deep_data_treatment(df):
    """进一步的特征工程：特征选择，特征构造"""
    return df


def model_param_tuning(train_x, train_y, test_x, test_y):
    """模型调参"""
    grid_search_param = [{
        'penalty': ["l1", 'l2'],
        'C': [0.1, 0.2, 0.5, 0.55, 0.6, 0.7, 1, 5, 10],
        'solver': ["liblinear"],
    },
        {
            'penalty': ['l2'],
            'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
            'solver': ["lbfgs"],
        }]
    grid_search_clf = GridSearchCV(linear_model.LogisticRegression(tol=1e-6, max_iter=1000),
                                   grid_search_param,
                                   cv=10,
                                   n_jobs=-1,)
    grid_search_clf.fit(train_x, train_y)
    print(grid_search_clf.best_params_)
    # print(grid_search_clf.best_estimator_.coef_)  # 逻辑回归的各项（特征）系数 == 特征重要性
    feature_importance = pd.DataFrame(grid_search_clf.best_estimator_.coef_.reshape(-1, 1),
                                      columns=['Feature_importance'],
                                      index=train_x.columns)
    print("{:*^30}".format("特征重要性排序"))
    print(feature_importance.sort_values(by='Feature_importance', ascending=False))
    feature_importance.sort_values(by='Feature_importance', ascending=False).plot(kind='barh')
    plt.show()

    print("{:*^30}".format("最佳模型评估效果"))
    print(cr(test_y, grid_search_clf.best_estimator_.predict(test_x)))


def plot_learning_curve(estimator, title, x, y, ylim=None, cv=5, n_jobs=1, train_sizes=np.linspace(.05, 1., 20),
                        verbose=0, plot=True):
    """
    画出data在某模型上的learning curve. 判断当前模型的状态：过拟合，欠拟合
    ----------
    estimator : 学习器
    title : 图像的标题
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
                         validation_scores_mean + validation_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, validation_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (
            validation_scores_mean[-1] - validation_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (validation_scores_mean[-1] - validation_scores_std[-1])
    return midpoint, diff


if __name__ == '__main__':
    data = get_data("train_set.csv")
    # print(data.head())

    df = basic_data_treatment(data)
    # eda_data(data)

    train_x, test_x, train_y, test_y = split_data(df, 'y')
    model_param_tuning(train_x, train_y, test_x, test_y)

    # plot_learning_curve(est, u"学习曲线", x, y)  # 绘制学习曲线
