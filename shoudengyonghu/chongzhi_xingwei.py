# -*- encoding: utf-8 -*-
# @Version : 1.0  
# @Time    : 2019/7/29 10:01
# @Author  : wanghd
# @note    :

import os
import re
import subprocess
import time
import statistics

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report as cr
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from figure_utils import df_box_plot, df_barplot, df_pair_boxplot, df_pair_plot
from figure_utils import plot_learning_curve, plot_validation_curve
from feature_engineer_utils import CustomDummifier, CustomEncoder, my_pipeline, reduce_mem_usage

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
plt.rcParams['font.family'] = 'sans-serif'  # 解决负号'-'显示为方块的问题
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


def write_data(df, filedir=os.getcwd(), filename='chongzhi', index=False):
    if not os.path.exists(filedir):
        os.mkdir(filedir)
    current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
    filename = f'{filename}_{current_time}.csv'
    df.to_csv(os.path.join(filedir, filename), index=index, header=True, encoding='gbk')


def split_data(df, label_col_name, test_size=0.2):
    label = df.pop(label_col_name)
    train_x, test_x, train_y, test_y = train_test_split(df, label, random_state=11, stratify=label, test_size=test_size)
    return train_x, test_x, train_y, test_y, df, label


def calculate_cards_value(df, cards_col, rank_col):
    """调用牌力值程序得到牌力值"""
    base_path = os.path.abspath("F:/CardsValue")
    input_file = os.path.join(base_path, 'Input.txt')
    if os.path.exists(input_file):
        os.remove(input_file)
    output_file = os.path.join(base_path, 'Output.txt')
    if os.path.exists(output_file):
        os.remove(output_file)
    # 写入牌面信息 到 input
    with open(input_file, 'a') as f:
        previous_card = ''  # 用于记录上轮出牌的临时变量，以便于排除不出牌
        for rowid in range(df.shape[0]):
            if df.at[rowid, "玩家ID"] and df.at[rowid, "玩家ID"] != previous_card:
                # 排除出完牌以及不出牌的情况
                f.write(str(int(df.at[rowid, "玩家ID"])))
                basic_list = [-1] * (33 - len(df.at[rowid, cards_col]))
                card_list = list(df.at[rowid, cards_col])
                card_list.extend(basic_list)
                f.write(os.linesep)
                # 将级牌拼接到牌组后面
                cards_str = ','.join([str(cards_id) for cards_id in card_list])
                try:
                    current_rank = str(int(df.at[rowid, rank_col]))
                except ValueError:
                    current_rank = '1'
                cards_str = f'{cards_str}||{current_rank}'
                f.write(cards_str)
                f.write(os.linesep)
                previous_card = df.at[rowid, "玩家ID"]

    # 执行 CalculatorCardsValue.exe 获得拆牌信息
    subprocess.call(os.path.join(base_path, 'CalculatorCardsValue1dir.exe'))

    # 读取拆牌信息
    df_cardsvalue_cols = ["玩家ID", 'cards_value', 'cards_id', 'cards_type']  # 唯一标识，牌力值，ID组，类型组
    # 加速读取速度
    df_cardsvalue = pd.read_csv(output_file, header=None, names=["result_abd"])
    if df_cardsvalue.shape[0] % 4 == 0:
        df_cardsvalue.loc[:, "mark"] = df_cardsvalue.index // 4  # 验证结果数目
        df_cardsvalue = df_cardsvalue.groupby('mark').apply(lambda x: x.T)  # 转置分组后的 series，mark列会变成行
        df_cardsvalue = df_cardsvalue.loc[(slice(None), "result_abd"), :].reset_index(drop=True)  # 筛出结果，去除mark行
        # df_cardsvalue.drop(columns=list(df_cardsvalue.columns)[-2], inplace=True)  # 删除结果中ID组列
        df_cardsvalue.columns = df_cardsvalue_cols
        df_cardsvalue.loc[:, 'cards_type'] = df_cardsvalue.loc[:, 'cards_type'].apply(
            lambda x: str(x).replace("-", "|"))
    else:
        df_cardsvalue = pd.DataFrame(columns=df_cardsvalue_cols)
        # 不满足结果数目的情况，使用 readline 保证完整记录可被记录
        with open(output_file, 'r') as fout:
            out = fout.readlines()
            out = eval(str(out).replace('-', '|'))
            for index, line in enumerate(out, start=1):
                if index % 5 == 1:
                    df_cardsvalue.at[index - 1, "玩家ID"] = line.strip()
                if index % 5 == 3:
                    df_cardsvalue.at[index - 3, 'cards_value'] = line.strip()
                if index % 5 == 4:
                    df_cardsvalue.at[index - 4, 'cards_id'] = line.strip()
                if index % 5 == 0:
                    df_cardsvalue.at[index - 5, 'cards_type'] = line.strip()
                # print(index, line)

    df_cardsvalue.drop_duplicates(inplace=True)

    df.loc[:, "玩家ID"] = df.loc[:, "玩家ID"].astype(int)
    df_cardsvalue.loc[:, "玩家ID"] = df_cardsvalue.loc[:, "玩家ID"].astype(int)
    df = pd.merge(df, df_cardsvalue, on=["玩家ID"], how='left', copy=False)  # 合并牌力值拆牌信息
    return df.loc[:, ["玩家ID", 'cards_value']]


def perform_calculated_cards_value(df):
    """执行牌力值程序计算牌力值, 拿到倒数第一局和倒数第二局的起手牌牌力值"""
    # 倒数第一局 级牌和起始手牌转化为 牌力值
    df_first_not_null = df.loc[df.loc[:, "倒数第一局起始手牌"].notnull()].reset_index(drop=True)
    df_first_not_null.loc[:, '倒数第一局起始手牌'] = df_first_not_null.loc[:, "倒数第一局起始手牌"].apply(
        lambda x: set(str(x).strip().split(sep=',')))

    df_first_not_null = calculate_cards_value(df_first_not_null, '倒数第一局起始手牌', '倒数第一局级牌')
    df_first_not_null.rename(columns={"cards_value": "倒数第一局牌力值"}, inplace=True)
    df = pd.merge(df, df_first_not_null, how='left', on='玩家ID')

    # 倒数第二局 级牌和起始手牌转化为 牌力值
    df_second_not_null = df.loc[df.loc[:, "倒数第二局起始手牌"].notnull()].reset_index(drop=True)
    df_second_not_null.loc[:, '倒数第二局起始手牌'] = df_second_not_null.loc[:, "倒数第二局起始手牌"].apply(
        lambda x: set(str(x).strip().split(sep=',')))

    df_second_not_null = calculate_cards_value(df_second_not_null, '倒数第二局起始手牌', '倒数第二局级牌')
    df_second_not_null.rename(columns={"cards_value": "倒数第二局牌力值"}, inplace=True)
    df = pd.merge(df, df_second_not_null, how='left', on='玩家ID')

    df.to_excel("掼蛋首登用户首次充值对局行为20190801plus牌力值.xlsx", index=False)
    return df


def eda_data(df):
    """初步的数据探索"""
    # 先查看数据基本情况
    df = df.replace("unknown", np.nan)

    # 缺失比例，均值，标准差，分位数等
    df_info = pd.DataFrame(df.isnull().sum(axis=0), columns=["nan_nums"])
    df_info.loc[:, 'nan_percent'] = df_info.loc[:, 'nan_nums'] / df.shape[0]
    df_info.loc[:, 'nunique'] = df.apply(pd.Series.nunique)  # 各列不同值的数量
    df_info = pd.merge(df_info, df.describe().T, left_index=True, right_index=True, how='left')
    pd.set_option("display.max_columns", 30)
    write_data(df_info, filename='chongzhi_df_info', index=True)
    print(df_info)

    # # 初步数据探索
    # # 数值型单变量分布
    # df_box_plot(df.loc[:, ['age', 'balance', 'day']], sub_title='')
    # # 离散型单变量分布
    # df_barplot(df, sub_title='')
    # # 离散型+ 数值型 单变量分布
    # df_pair_boxplot(df, 'housing', ['age', ], 'y', figsize=(12, 16))
    # # 数值型双变量相关性
    # df_pair_plot(df.loc[:, ['age', 'balance', 'day', 'job', 'y']])
    return None


def basic_data_treatment(df):
    """初步数据处理（预处理）"""

    # 提取时间列的信息
    time_cols = ["充值时间", "倒数第一局对局时间", "倒数第二局对局时间"]
    time_point_cols = [f"{time_col}点" for time_col in time_cols]
    time_inter_cols = [f"{time_col}段" for time_col in time_cols]
    for time_col, time_point_col, time_inter_col in zip(time_cols, time_point_cols, time_inter_cols):
        data_not_None = df.loc[~df.loc[:, time_col].isnull()].reset_index(drop=True)
        # 提取时间中的小时
        data_not_None.loc[:, time_point_col] = data_not_None.loc[:, time_col].apply(
            lambda x: re.match("[\d+(\W+]+T(\d+):[\W+|\d+]+", str(x)).groups()[0])
        data_not_None.loc[:, time_point_col] = data_not_None.loc[:, time_point_col].astype(int)
        # 将小时转为是时间段["0-8点", "8-12点", "12-18点", "18-24点"]
        data_not_None.loc[:, time_inter_col] = pd.cut(data_not_None.loc[:, time_point_col],
                                                      bins=[0, 8, 12, 18, 24],
                                                      right=False,
                                                      include_lowest=True,
                                                      labels=["0-8点", "8-12点", "12-18点", "18-24点"]
                                                      )
        df = pd.merge(df, data_not_None.loc[:, ["玩家ID", time_point_col, time_inter_col]],
                      on="玩家ID",
                      how="left")

    # df.loc[:, '是否平台首充'] = df.loc[:, '是否平台首充'].map({"TRUE": 1, "False": 0})

    # 倒数两局是否存在房间变化
    df.loc[:, "是否倒数两局存在房间变化"] = df.apply(
        lambda row: 1 if row["倒数第二局房间号"] != row["倒数第二局房间号"] else 0, axis=1)

    # pd.set_option('display.max_columns', 30)
    return df


def deep_data_treatment(df):
    """进一步的特征工程：缺失值填充，特征选择，特征构造"""
    # 类别型热编码
    dummy = CustomDummifier(cols=["倒数第一局房间号", "倒数第二局房间号", "倒数第一局对局时间段",
                                  '倒数第二局对局时间段'])
    df = dummy.fit_transform(df)

    # 填充缺失值为 0 的列
    fillna_cols = ["首登是否当天充值", '是否平台首充', '充值金额', "首充前当日对局数", "倒数第一局对局时间",
                   "倒数第一局房间号", "倒数第二局对局时间", "倒数第二局房间号"]
    for col in fillna_cols:
        df.loc[:, col] = df.loc[:, col].fillna(0)
    return df


def basic_model_selection(basic_models=None, x=None, y=None, scoring='roc_auc', cv=5):
    """初步的算法筛选，确定一个或者几个基学习器进行下一步调参"""
    if basic_models is None:
        all_models = [linear_model.LogisticRegression(),
                      KNeighborsClassifier(),
                      DecisionTreeClassifier(),
                      # SVC(),
                      AdaBoostClassifier(),
                      AdaBoostClassifier(learning_rate=0.5),
                      AdaBoostClassifier(base_estimator=linear_model.LogisticRegression()),
                      RandomForestClassifier(),
                      GradientBoostingClassifier()]
    else:
        all_models = basic_models
    cv_scores = []
    for basic_model in all_models:
        cv_scores.append(cross_val_score(basic_model, X=x, y=y, scoring=scoring, cv=cv))
    cv_score_df = pd.DataFrame(cv_scores).T  # 获得各个学习器每折的score
    cv_score_df.columns = [basic_model.__class__.__name__ for basic_model in all_models]  # 将列名命名为学习器名
    cv_score_df.index = [f"cv_{cv_round + 1}" for cv_round in cv_score_df.index]  # index命名为cv_{index_round}
    cv_score_df = pd.concat([cv_score_df, cv_score_df.describe()], axis=0)  # 加入各轮cv的scores的统计信息
    print(cv_score_df)
    write_data(cv_score_df, filename='basic_model_selection', index=True)
    return cv_score_df


def model_param_tuning_lr(train_x, train_y, test_x, test_y, scoring='roc_auc', grid_param=None):
    """模型调参"""
    if grid_param:
        grid_search_param = grid_param.copy()
    else:
        grid_search_param = [{
            'penalty': ["l1", 'l2'],
            'C': [0.01, 0.1, 1, 5, 6, 7, 10],
            'solver': ["liblinear"],
        },
            # {
            #     'penalty': ['l2'],
            #     'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
            #     'solver': ["lbfgs"],
            # }
        ]
    grid_search_clf = GridSearchCV(linear_model.LogisticRegression(tol=1e-6, max_iter=1000),
                                   grid_search_param,
                                   cv=10,
                                   n_jobs=-1,
                                   scoring=scoring)
    grid_search_clf.fit(train_x, train_y)
    print(grid_search_clf.best_params_)
    # print(grid_search_clf.best_estimator_.coef_)  # 逻辑回归的各项（特征）系数 == 特征重要性
    feature_importance = pd.DataFrame(grid_search_clf.best_estimator_.coef_.reshape(-1, 1),
                                      columns=['Feature_importance'],
                                      index=train_x.columns)
    all_importances = abs(feature_importance.loc[:, 'Feature_importance']).sum()
    # 特征重要型占比
    feature_importance.loc[:, "feature_importance_percent"] = feature_importance.loc[:,
                                                              'Feature_importance'] / all_importances
    # 百分比都表示为 正 值
    feature_importance.loc[:, 'normalized_importance'] = feature_importance.feature_importance_percent.apply(abs)
    # 输出百分比格式列
    feature_importance.loc[:, 'normalized_importance_percentile'] = feature_importance.normalized_importance.apply(
        lambda x: "{:.2%}".format(x)
    )
    write_data(feature_importance, filename='lr_feature_importance', index=True)
    print("{:*^30}".format("lr特征重要性排序"))
    print(feature_importance.sort_values(by='Feature_importance', ascending=False))
    feature_importance.Feature_importance.sort_values(ascending=False).plot(kind='barh')
    feature_importance.normalized_importance.sort_values(ascending=False).plot(kind='barh')
    plt.show()

    print("{:*^30}".format("lr最佳模型评估效果"))
    print(cr(test_y, grid_search_clf.best_estimator_.predict(test_x)))

    print("{:*^30}".format("lr最佳模型auc值"))
    print(roc_auc_score(test_y, grid_search_clf.best_estimator_.predict_proba(test_x)[:, 1]))

    print("{:*^30}".format("lr最佳accuracy"))
    print(accuracy_score(test_y, grid_search_clf.best_estimator_.predict(test_x)))
    print(grid_search_clf.best_score_)

    return grid_search_clf.best_estimator_


def model_param_tuning_gbdt(train_x, train_y, test_x, test_y, scoring='roc_auc', grid_param=None):
    """GBDT模型调参"""
    # {'learning_rate': 0.1, 'max_depth': 7, 'min_samples_leaf': 80, 'min_samples_split': 150, 'n_estimators': 45}
    if grid_param:
        grid_search_param = grid_param.copy()
    else:
        grid_search_param = [{
            'learning_rate': [0.001, 0.01, 0.1, 0.3, 0.5, 0.8],
            'n_estimators': range(20, 101, 10),
            'max_depth': range(3, 22, 2),
            'min_samples_split': range(100, 801, 200),
            'min_samples_leaf': range(60, 101, 10)
        },
        ]
    grid_search_clf = GridSearchCV(GradientBoostingClassifier(tol=1e-6),
                                   grid_search_param,
                                   cv=10,
                                   n_jobs=-1,
                                   scoring=scoring)
    grid_search_clf.fit(train_x, train_y)
    print(grid_search_clf.best_params_)
    feature_importance = pd.DataFrame(grid_search_clf.best_estimator_.feature_importances_.reshape(-1, 1),
                                      columns=['Feature_importance'],
                                      index=train_x.columns)
    all_importances = abs(feature_importance.loc[:, 'Feature_importance']).sum()
    # 特征重要型占比
    feature_importance.loc[:, "feature_importance_percent"] = feature_importance.loc[:,
                                                              'Feature_importance'] / all_importances
    # 百分比都表示为 正 值
    feature_importance.loc[:, 'normalized_importance'] = feature_importance.feature_importance_percent.apply(abs)
    # 输出百分比格式列
    feature_importance.loc[:, 'normalized_importance_percentile'] = feature_importance.normalized_importance.apply(
        lambda x: "{:.2%}".format(x)
    )
    write_data(feature_importance, filename='gbdt_feature_importance',index=True)
    print("{:*^30}".format("gbdt特征重要性排序"))
    print(feature_importance.sort_values(by='Feature_importance', ascending=False))
    feature_importance.Feature_importance.sort_values(ascending=False).plot(kind='barh')
    plt.show()

    print("{:*^30}".format("gbdt最佳模型评估效果"))
    print(cr(test_y, grid_search_clf.best_estimator_.predict(test_x)))

    print("{:*^30}".format("gbdt最佳模型auc值"))
    print(roc_auc_score(test_y, grid_search_clf.best_estimator_.predict_proba(test_x)[:, 1]))

    print("{:*^30}".format("gbdt最佳accuracy"))
    print(accuracy_score(test_y, grid_search_clf.best_estimator_.predict(test_x)))
    return grid_search_clf.best_estimator_


def plat_recharge_classfication():
    """平台用户在掼蛋再充值的影响因素/ 平台用户在掼蛋首充的影响因素"""
    pd.set_option("display.max_columns", 30)
    # data_cz = get_data("掼蛋首登用户首次充值对局行为20190801.xlsx",)
    # 计算牌力值
    # data_cz = perform_calculated_cards_value(data_cz)
    #
    # data_cz = get_data("掼蛋首登用户首次充值对局行为20190801plus牌力值.xlsx",)
    # data_cz = basic_data_treatment(data_cz)  # 特征提取
    ## write_data(data_cz, filename='掼蛋首登用户首次充值对局行为20190801plus牌力值等.xlsx')
    # write_data(data_cz, filename='chongzhi')
    # data_cz = reduce_mem_usage(data_cz, verbose=True)

    data_cz = pd.read_csv("chongzhi.csv", encoding='gbk')
    # print(data_cz.columns)
    # eda_data(data_cz)
    #
    # 充值用户= 平台首充+非平台首充  label: 是否平台首充
    data_recharge = data_cz.loc[~data_cz.loc[:, "充值时间"].isnull()].reset_index(drop=True)
    # 填充首充前的数据 用0填充
    for col in [col for col in data_recharge.columns if col.startswith("首充前")]:
        data_recharge.loc[:, col].fillna(0, inplace=True)
    # 填充时间点和时间段
    for col in ["倒数第一局对局时间点", '倒数第二局对局时间点', '倒数第一局对局时间段', '倒数第二局对局时间段']:
        data_recharge.loc[:, col].fillna(statistics.mode(data_recharge.loc[:, col]), inplace=True)
    # 房间号无需填充，
    # -1填充
    na_cols = ["倒数第一局是否房间破产", "倒数第二局是否房间破产", "倒数第一局跑牌顺序", "倒数第二局跑牌顺序",
               "倒数第一局级牌", "倒数第二局级牌", "倒数第一局牌力值", "倒数第二局牌力值"]
    for col in na_cols:
        data_recharge.loc[:, col] = data_recharge.loc[:, col].fillna(-1)
    # 倍数填充
    beishu_cols = ["倒数第一局倍数", "倒数第二局倍数"]
    for beishu_col in beishu_cols:
        data_recharge.loc[:, beishu_col] = data_recharge.loc[:, beishu_col].fillna(-1)

    # 删除无需列
    data_recharge.drop(columns=["充值商品", "倒数第一局对局标识", "倒数第二局对局标识"], inplace=True)
    # df.drop(columns=["充值商品"], inplace=True)
    data_recharge.drop(columns=["首登日期"], inplace=True)  # 暂时先删除
    data_recharge.drop(columns=["首充日期"], inplace=True)  # 暂时先删除
    data_recharge.drop(columns=["玩家ID"], inplace=True)
    data_recharge.drop(columns=["充值时间", '倒数第一局对局时间', '倒数第二局对局时间'], inplace=True)
    data_recharge.drop(columns=['倒数第一局起始手牌', '倒数第二局起始手牌'], inplace=True)
    data_recharge.drop(columns=['次日是否留存'], inplace=True)

    # one hot
    discrete_cols = ["倒数第一局房间号", '倒数第二局房间号', '倒数第一局对局时间段', '倒数第二局对局时间段', '充值时间段']
    dummy = CustomDummifier(cols=discrete_cols)
    data_recharge = dummy.fit_transform(data_recharge)

    # eda_data(data_recharge)

    # 算法选择
    print(data_recharge.loc[:, "是否平台首充"].value_counts().max() / data_recharge.shape[0])
    train_x, test_x, train_y, test_y, chunk_df, label = split_data(data_recharge, '是否平台首充')
    # basic_model_selection(basic_models=None, x=train_x, y=train_y, scoring='accuracy', cv=10)
    # lr, adaboost, gbdt

    # 调参
    lr_param = [{
        'penalty': ["l1", 'l2'],
        'C': [0.01, 0.05, 0.1, 1, 5, 6, 7, 10],
        'solver': ["liblinear"],
    }]
    best_est = model_param_tuning_lr(train_x, train_y, test_x, test_y, scoring='accuracy',
                                     grid_param=lr_param)
    best_est.fit(chunk_df, label)
    feature_importance = pd.DataFrame(best_est.coef_.reshape(-1, 1),
                                      columns=['Feature_importance'],
                                      index=train_x.columns)
    all_importances = abs(feature_importance.loc[:, 'Feature_importance']).sum()
    # 特征重要型占比
    feature_importance.loc[:, "feature_importance_percent"] = feature_importance.loc[:,
                                                              'Feature_importance'] / all_importances
    # 百分比都表示为 正 值
    feature_importance.loc[:, 'normalized_importance'] = feature_importance.feature_importance_percent.apply(abs)
    # 输出百分比格式列
    feature_importance.loc[:, 'normalized_importance_percentile'] = feature_importance.normalized_importance.apply(
        lambda x: "{:.2%}".format(x)
    )
    write_data(feature_importance, filename='lr_feature_importance_alldf',index=True)

    gbdt_param = [{
        'learning_rate': [0.001, 0.01, 0.1, 0.3, 0.5, 0.8, 1, 3],
        'n_estimators': range(10, 50, 5),
        'max_depth': range(5, 15, 1),
        'min_samples_split': range(150, 350, 50),
        'min_samples_leaf': range(70, 101, 10)
    }]
    best_gbdt = model_param_tuning_gbdt(train_x, train_y, test_x, test_y, scoring='accuracy',
                                     grid_param=gbdt_param)
    best_gbdt.fit(chunk_df, label)
    feature_importance = pd.DataFrame(best_gbdt.feature_importances_.reshape(-1, 1),
                                      columns=['Feature_importance'],
                                      index=train_x.columns)
    all_importances = abs(feature_importance.loc[:, 'Feature_importance']).sum()
    # 特征重要型占比
    feature_importance.loc[:, "feature_importance_percent"] = feature_importance.loc[:,
                                                              'Feature_importance'] / all_importances
    # 百分比都表示为 正 值
    feature_importance.loc[:, 'normalized_importance'] = feature_importance.feature_importance_percent.apply(abs)
    # 输出百分比格式列
    feature_importance.loc[:, 'normalized_importance_percentile'] = feature_importance.normalized_importance.apply(
        lambda x: "{:.2%}".format(x)
    )
    write_data(feature_importance, filename='gbdt_feature_importance_alldf', index=True)

    # # 判断模型状态
    # plot_learning_curve(AdaBoostClassifier(), u"学习曲线", train_x, train_y, ylim=(0.5, 0.9))  # 绘制学习曲线
    # plot_learning_curve(AdaBoostClassifier(), u"学习曲线", chunk_df, label, ylim=(0.5, 0.9))  # 绘制学习曲线

    # 验证曲线
    # plot_validation_curve(linear_model.LogisticRegression(), chunk_df, label,
    #                       'C', range(1,10),
    #                       )
    # plot_validation_curve(GradientBoostingClassifier(), chunk_df, label,
    #                       # 'learning_rate', range(1,5),
    #                       # 'n_estimators', range(10, 101, 5), #[10-20]
    #                       # 'max_depth', range(3, 22, 2),  # [8-10]
    #                       # 'min_samples_split', range(100, 801, 50), # [200-300]
    #                       'min_samples_leaf', range(60, 101, 10),  # [70,]
    #                       y_lim=(0.6,1.1)
    #                       )


def whether_charge_classfication(model_filter=True):
    """是否充值影响因素"""
    pd.set_option("display.max_columns", 30)
    # data_cz = get_data("掼蛋首登用户首次充值对局行为20190801.xlsx",)
    # 计算牌力值
    # data_cz = perform_calculated_cards_value(data_cz)
    #
    # data_cz = get_data("掼蛋首登用户首次充值对局行为20190801plus牌力值.xlsx",)
    # data_cz = basic_data_treatment(data_cz)  # 特征提取
    ## write_data(data_cz, filename='掼蛋首登用户首次充值对局行为20190801plus牌力值等.xlsx')
    # write_data(data_cz, filename='chongzhi')
    # data_cz = reduce_mem_usage(data_cz, verbose=True)

    data_cz = pd.read_csv("chongzhi.csv", encoding='gbk')
    # 加入部分用户基本信息
    data_user_info = pd.read_excel("掼蛋用户基础信息.xlsx", usecols=["玩家ID", '手机系统', '包体ID', '登陆天数'])
    data_user_info.loc[:, "包体ID"] =data_user_info.loc[:, "包体ID"].fillna(999)
    data_user_info.loc[:, "包体ID"] =data_user_info.loc[:, "包体ID"].astype(str)
    # 将包体ID one hot
    data_user_info_dummy = CustomDummifier(cols=["包体ID"])
    data_user_info = data_user_info_dummy.fit_transform(data_user_info)
    # data_user_info = data_user_info.loc[
    #     data_user_info.loc[:, '玩家ID'].isin(data_cz.loc[:, '玩家ID'].unique())].reset_index(drop=True)
    data_cz = pd.merge(data_cz, data_user_info, on=['玩家ID'], how='left')
    print(data_cz.columns)
    # eda_data(data_cz)

    # label: 是否充值
    data_cz.loc[:, "充值金额"] = data_cz.loc[:, "充值金额"].fillna(0)
    data_cz.loc[:, "充值金额"] = data_cz.loc[:, "充值金额"].apply(lambda x: 1 if x > 0 else 0)
    data_cz.rename(columns={"充值金额": "是否充值"}, inplace=True)

    # 首充前的数据,充值相关的数据，删除
    data_cz.drop(columns=[col for col in data_cz.columns if col.startswith("首充前")], inplace=True)
    data_cz.drop(columns=["充值时间", "充值时间段", "充值时间点"], inplace=True)
    data_cz.drop(columns=["首登是否当天充值", "是否平台首充"], inplace=True)

    # 填充时间点和时间段
    for col in ["倒数第一局对局时间点", '倒数第二局对局时间点', '倒数第一局对局时间段', '倒数第二局对局时间段']:
        data_cz.loc[:, col].fillna(statistics.mode(data_cz.loc[:, col]), inplace=True)
    # 房间号无需填充，
    # -1填充
    na_cols = ["倒数第一局是否房间破产", "倒数第二局是否房间破产", "倒数第一局跑牌顺序", "倒数第二局跑牌顺序",
               "倒数第一局级牌", "倒数第二局级牌", "倒数第一局牌力值", "倒数第二局牌力值"]
    for col in na_cols:
        data_cz.loc[:, col] = data_cz.loc[:, col].fillna(-1)
    # 倍数填充
    beishu_cols = ["倒数第一局倍数", "倒数第二局倍数"]
    for beishu_col in beishu_cols:
        data_cz.loc[:, beishu_col] = data_cz.loc[:, beishu_col].fillna(-1)

    # 删除无需列
    # data_cz.drop(columns=["充值商品", "倒数第一局对局标识", "倒数第二局对局标识"], inplace=True)
    # data_cz.drop(columns=["首登日期"], inplace=True)
    # data_cz.drop(columns=["首充日期"], inplace=True)
    data_cz.drop(columns=["玩家ID"], inplace=True)
    data_cz.drop(columns=['倒数第一局对局时间', '倒数第二局对局时间'], inplace=True)
    data_cz.drop(columns=['倒数第一局起始手牌', '倒数第二局起始手牌'], inplace=True)
    data_cz.drop(columns=['次日是否留存'], inplace=True)

    # one hot
    discrete_cols = ["倒数第一局房间号", '倒数第二局房间号', '倒数第一局对局时间段', '倒数第二局对局时间段']
    dummy = CustomDummifier(cols=discrete_cols)
    data_cz = dummy.fit_transform(data_cz)

    eda_data(data_cz)

    # 算法选择
    print(data_cz.loc[:, "是否充值"].value_counts().max() / data_cz.shape[0])
    train_x, test_x, train_y, test_y, chunk_df, label = split_data(data_cz, '是否充值')
    if model_filter:
        basic_model_selection(basic_models=None, x=train_x, y=train_y, scoring='accuracy', cv=10)
        # lr, adaboost, gbdt
    else:
        # 调参
        lr_param = [{
            'penalty': ["l1", 'l2'],
            'C': [0.01, 0.05, 0.1, 1, 5, 6, 7, 10],
            'solver': ["liblinear"],
        }]
        lr_param = [{'C': [5], 'penalty': ['l1'], 'solver': ['liblinear']}]  # 寻参得到的新参数
        best_est = model_param_tuning_lr(train_x, train_y, test_x, test_y, scoring='accuracy',
                                         grid_param=lr_param)
        best_est.fit(chunk_df, label)
        feature_importance = pd.DataFrame(best_est.coef_.reshape(-1, 1),
                                          columns=['Feature_importance'],
                                          index=train_x.columns)
        all_importances = abs(feature_importance.loc[:, 'Feature_importance']).sum()
        # 特征重要型占比
        feature_importance.loc[:, "feature_importance_percent"] = feature_importance.loc[:,
                                                                  'Feature_importance'] / all_importances
        # 百分比都表示为 正 值
        feature_importance.loc[:, 'normalized_importance'] = feature_importance.feature_importance_percent.apply(abs)
        # 输出百分比格式列
        feature_importance.loc[:, 'normalized_importance_percentile'] = feature_importance.normalized_importance.apply(
            lambda x: "{:.2%}".format(x)
        )
        write_data(feature_importance, filename='lr_feature_importance_alldf',index=True)

        gbdt_param = [{
            'learning_rate': [0.001, 0.01, 0.1, 0.3, 0.5, 0.8, 1, 3],
            'n_estimators': range(10, 50, 5),
            'max_depth': range(5, 15, 1),
            'min_samples_split': range(150, 350, 50),
            'min_samples_leaf': range(70, 101, 10)
        }]
        gbdt_param = [{'learning_rate': [0.1], 'max_depth': [5], 'min_samples_leaf': [80], 'min_samples_split': [250],
                       'n_estimators': [45]}]   # 寻参得到的新参数
        best_gbdt = model_param_tuning_gbdt(train_x, train_y, test_x, test_y, scoring='accuracy',
                                         grid_param=gbdt_param)
        best_gbdt.fit(chunk_df, label)
        feature_importance = pd.DataFrame(best_gbdt.feature_importances_.reshape(-1, 1),
                                          columns=['Feature_importance'],
                                          index=train_x.columns)
        all_importances = abs(feature_importance.loc[:, 'Feature_importance']).sum()
        # 特征重要型占比
        feature_importance.loc[:, "feature_importance_percent"] = feature_importance.loc[:,
                                                                  'Feature_importance'] / all_importances
        # 百分比都表示为 正 值
        feature_importance.loc[:, 'normalized_importance'] = feature_importance.feature_importance_percent.apply(abs)
        # 输出百分比格式列
        feature_importance.loc[:, 'normalized_importance_percentile'] = feature_importance.normalized_importance.apply(
            lambda x: "{:.2%}".format(x)
        )
        write_data(feature_importance, filename='gbdt_feature_importance_alldf', index=True)

        # # 判断模型状态
        # plot_learning_curve(AdaBoostClassifier(), u"学习曲线", train_x, train_y, ylim=(0.5, 0.9))  # 绘制学习曲线
        # plot_learning_curve(AdaBoostClassifier(), u"学习曲线", chunk_df, label, ylim=(0.5, 0.9))  # 绘制学习曲线

        # 验证曲线
        # plot_validation_curve(linear_model.LogisticRegression(), chunk_df, label,
        #                       'C', range(1,10),
        #                       )
        # plot_validation_curve(GradientBoostingClassifier(), chunk_df, label,
        #                       # 'learning_rate', range(1,5),
        #                       # 'n_estimators', range(10, 101, 5), #[10-20]
        #                       # 'max_depth', range(3, 22, 2),  # [8-10]
        #                       # 'min_samples_split', range(100, 801, 50), # [200-300]
        #                       'min_samples_leaf', range(60, 101, 10),  # [70,]
        #                       y_lim=(0.6,1.1)
        #                       )


if __name__ == '__main__':
    # plat_recharge_classfication()
    whether_charge_classfication(model_filter=False)
