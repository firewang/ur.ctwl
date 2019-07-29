# -*- encoding: utf-8 -*-
# @Version : 1.0  
# @Time    : 2019/7/29 15:12
# @Author  : wanghd
# @note    :

from math import sqrt, ceil
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# sns.set(font="SimHei")


def df_box_plot(df, layout=None, sub_title='box_plot'):
    """数值型变量（单变量）分布情况"""
    plt.figure(1, figsize=(12, 8))
    for i, col in enumerate(df.columns):
        if layout is None:
            # 如果不指定 layout，则根据列数开方进行计算指定
            squared_layout = ceil(sqrt(df.shape[1]))
            plt.subplot(squared_layout, squared_layout, i + 1)
        else:
            plt.subplot(*layout, i + 1)
        plt.boxplot(df.loc[:, col])
        plt.title(col)
    plt.suptitle(sub_title)
    plt.show()


def df_barplot(df, layout=None, sub_title="bar_plot"):
    """离散型变量（单变量）分布情况"""
    plt.figure(1, figsize=(12, 6))
    for i, col in enumerate(df.columns):
        if layout is None:
            # 如果不指定 layout，则根据列数开方进行计算指定
            squared_layout = ceil(sqrt(df.shape[1]))
            plt.subplot(squared_layout, squared_layout, i + 1)
        else:
            plt.subplot(*layout, i + 1)
        count = df[col].value_counts()
        plt.bar(count.index, count.values, width=0.5)
        plt.title(col)
    plt.suptitle(sub_title)
    plt.show()


def df_pair_boxplot(df, x, y, hue, layout=None, figsize=(12, 16), sub_title='pair_boxplot'):
    """分组箱线图，横轴x(类别型), 纵轴为y（可以为多列，数值型）, 根据hue分组（类别型）
    即查看数值型变量 在不同类别分组条件下的 分布情况
    离散型（单变量）和数值型（单变量）"""
    plt.figure(1, figsize=figsize, dpi=300)
    for i, col in enumerate(y):
        if layout is None:
            # 如果不指定 layout，则根据列数指定行数，列数总为1
            plt.subplot(len(y), 1, i + 1)
        else:
            plt.subplot(*layout, i + 1)
        sns.boxplot(x, col, hue, df)
    plt.suptitle(sub_title)
    plt.show()


def df_pair_plot(df):
    """数值型变量的相关分析图， 会自动过滤非数值型列
    数值型（双变量）"""
    sns.pairplot(df)
    plt.show()


if __name__ == '__main__':
    pass
