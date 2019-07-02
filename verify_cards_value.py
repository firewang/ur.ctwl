# -*- encoding: utf-8 -*-
# @Version : 1.0  
# @Time    : 2019/7/1 15:00
# @Author  : wanghd
# @note    :  验证牌力值，1。牌力值与跑牌顺序 ；2。 牌力值总和大的一方的胜率

import os
import time
import pandas as pd
import subprocess


def first_init():
    """初始化基础配置"""
    global rawdatadir
    global tmpdatadir
    global outdatadir
    global rawdatadir1
    global tmpdatadir1
    global outdatadir1

    rawdatadir = os.path.join(os.getcwd(), 'rawdata')
    tmpdatadir = os.path.join(os.getcwd(), 'tmpdata')
    outdatadir = os.path.join(os.getcwd(), 'outdata')

    rawdatadir1 = os.path.join(os.getcwd(), 'rawdata1')
    tmpdatadir1 = os.path.join(os.getcwd(), 'tmpdata1')
    outdatadir1 = os.path.join(os.getcwd(), 'outdata1')


def write_data(df, filedir='D:/projectsHome/ur.ctwl/tmpdata1', filename='aaaaaaa', index=False):
    current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
    filename = f'{filename}-{current_time}.csv'
    df.to_csv(os.path.join(filedir, filename), index=index, header=True)


def reduce_data():
    first_init()
    # 缩减customization 定制表体积
    cus_usecols = ['startguid', 'uid', 'cards', 'rank', 'winorder']  # 起手牌,级牌,跑牌顺序
    cus_files = [file for file in os.listdir(rawdatadir) if file.startswith('customization')]
    cus_files = cus_files[-1:]
    for cus_file in cus_files:
        cus = pd.read_csv(os.path.join(rawdatadir, cus_file), usecols=cus_usecols)
        cus.to_csv(os.path.join(rawdatadir, f"short_{cus_file}"), index=False, header=True)

    # 缩减showcards 出牌信息表体积
    show_usecols = ['startguid', 'uid', 'playtime_unix']  # 出牌时间
    show_files = [file for file in os.listdir(rawdatadir) if file.startswith('showcards')]
    show_files = show_files[-1:]
    for show_file in show_files:
        show = pd.read_csv(os.path.join(rawdatadir, show_file), usecols=show_usecols)
        show.to_csv(os.path.join(rawdatadir, f"short_{show_file}"), index=False, header=True)

    # 缩减start开局表，有历史对局战绩信息
    start_usecols = ['startguid', 'uid', 'win', 'loss']  # 历史胜局数，历史输局数
    start_files = [file for file in os.listdir(rawdatadir) if file.startswith('start')]
    start_files = start_files[-1:]
    for start_file in start_files:
        start = pd.read_csv(os.path.join(rawdatadir, start_file), usecols=start_usecols)
        start.to_csv(os.path.join(rawdatadir, f"short_{start_file}"), index=False, header=True)


def get_raw_data():
    """读取正式数据"""
    cus_usecols = ['startguid', 'uid', 'cards', 'rank', 'winorder']  # 起手牌,级牌,跑牌顺序
    cus_files = [file for file in os.listdir(rawdatadir) if file.startswith('short_customization_20190627')]
    name_dict = {'cards': "cards_init"}  # 重命名为 初始牌组

    # 对局id,uid, 开局时间，每手出牌时间点，出牌状态（出，不出，托管）'optype',
    # 牌型type，出牌回合, 出牌信息id[0-107]，出牌数字信息[1,14]
    # ['startguid', 'uid', 'starttime_unix', 'playtime_unix', 'type', 'cards_order', 'cards', 'num']
    show_usecols = ['startguid', 'uid', 'playtime_unix']  # 出牌时间
    show_files = [file for file in os.listdir(rawdatadir) if file.startswith('short_showcards_20190627')]

    start_usecols = ['startguid', 'uid', 'win', 'loss']  # 历史胜局数，历史输局数
    start_files = [file for file in os.listdir(rawdatadir) if file.startswith('short_start')]

    cus = pd.read_csv(os.path.join(rawdatadir, cus_files[-1]), usecols=cus_usecols)  # 读取定制信息
    cus.rename(columns=name_dict, inplace=True)  # 重命名为初始牌组
    # cus['cards_order'] = 1  # 起手牌标记出牌回合为 1
    show = pd.read_csv(os.path.join(rawdatadir, show_files[-1]), usecols=show_usecols)  # 读取出牌信息
    start = pd.read_csv(os.path.join(rawdatadir, start_files[-1]), usecols=start_usecols)  # 读取历史战绩

    return cus, show, start


def get_test_data():
    """读取测试数据"""
    cus_usecols = ['startguid', 'uid', 'cards', 'rank', 'winorder']  # 起手牌,级牌,跑牌顺序
    cus_files = [file for file in os.listdir(tmpdatadir1) if file.startswith('cus_test')]
    name_dict = {'cards': "cards_init"}  # 重命名为 初始牌组

    # 对局id,uid, 开局时间，每手出牌时间点，出牌状态（出，不出，托管）'optype',
    # 牌型type，出牌回合, 出牌信息id[0-107]，出牌数字信息[1,14]
    # ['startguid', 'uid', 'starttime_unix', 'playtime_unix', 'type', 'cards_order', 'cards', 'num']
    show_usecols = ['startguid', 'uid', 'playtime_unix']  # 出牌时间
    show_files = [file for file in os.listdir(tmpdatadir1) if file.startswith('show_test')]

    start_usecols = ['startguid', 'uid', 'win', 'loss']  # 历史胜局数，历史输局数
    start_files = [file for file in os.listdir(tmpdatadir1) if file.startswith('start_test')]

    cus = pd.read_csv(os.path.join(tmpdatadir1, cus_files[-1]), usecols=cus_usecols)  # 读取定制信息
    cus.rename(columns=name_dict, inplace=True)  # 重命名为初始牌组
    # cus['cards_order'] = 1  # 起手牌标记出牌回合为 1
    show = pd.read_csv(os.path.join(tmpdatadir1, show_files[-1]), usecols=show_usecols)  # 读取出牌信息
    start = pd.read_csv(os.path.join(tmpdatadir1, start_files[-1]), usecols=start_usecols)  # 读取历史战绩

    return cus, show, start


def judge_win_result(show_df, cus_df):
    # 构建座位号
    seat_order = show_df.loc[:, ['startguid', 'uid', 'playtime_unix']].sort_values(by=['startguid', 'playtime_unix'])
    seat_order = seat_order.groupby(["startguid"]).head(4)  # 仅取该局前4手牌
    seat_order.reset_index(drop=True, inplace=True)  # 重置index
    seat_order.loc[:, "seat_order"] = seat_order.index % 4 + 1
    seat_order.drop(columns=['playtime_unix'], inplace=True)
    seat_order.loc[:, "pair_seat_order"] = seat_order.loc[:, "seat_order"].apply(lambda x: x + 2 if x < 3 else x - 2)

    df = pd.merge(seat_order, cus_df, on=["startguid", 'uid'], copy=False)  # 合并定制表 和 出牌表
    pari_winorder_df = df.loc[:, ["startguid", 'seat_order', 'winorder']]
    pari_winorder_df.rename(columns={'winorder': "pair_winorder", "seat_order": "pair_seat_order"}, inplace=True)
    df = pd.merge(df, pari_winorder_df,
                  on=["startguid", 'pair_seat_order'],
                  )
    df.loc[:, "winorder"] = df.loc[:, "winorder"].astype(int)
    df.loc[:, "pair_winorder"] = df.loc[:, "pair_winorder"].astype(int)
    df.loc[:, "win_result"] = df.apply(lambda row: (row["winorder"], row["pair_winorder"]), axis=1)
    # 构建单扣，双扣，平扣的 胜负对照字典
    win_dict = {
        (1, 2): 3, (1, 3): 2, (1, 4): 1, (2, 3): -1, (2, 4): -2, (3, 4): -3, (4, 4): -3,
        (2, 1): 3, (3, 1): 2, (4, 1): 1, (3, 2): -1, (4, 2): -2, (4, 3): -3
    }
    # 将胜负基于 对照字典映射为 数值
    df.loc[:, 'win_result'] = df.loc[:, 'win_result'].map(win_dict)
    df.loc[:, "win_loss"] = df["win_result"].apply(lambda x: 1 if x > 0 else 0)
    # df.drop(columns = ["pair_winorder"],inplace=True)
    return df


def calculate_cards_value(df):
    """调用拆牌程序拆解剩余手牌"""
    base_path = "F:/CardsValue"
    inputfile = os.path.join(base_path, 'Input.txt')
    if os.path.exists(inputfile):
        os.remove(inputfile)
    outputfile = os.path.join(base_path, 'Output.txt')
    if os.path.exists(outputfile):
        os.remove(outputfile)

    # 构建牌组唯一标识
    df.loc[:, 'cards_init'] = df.loc[:, 'cards_init'].apply(lambda x: set(x.split(sep=',')))
    df.loc[:, 'leftcards_str'] = df.loc[:, 'cards_init'].apply(lambda x: "".join(sorted(list(x))))

    # 写入牌面信息 到 input
    with open(os.path.join(base_path, 'Input.txt'), 'a') as f:
        previous_card = ''  # 用于记录上轮出牌的临时变量，以便于排除不出牌
        for rowid in range(df.shape[0]):
            if df.at[rowid, 'leftcards_str'] and df.at[rowid, 'leftcards_str'] != previous_card:
                # 排除出完牌以及不出牌的情况
                f.write(df.at[rowid, 'leftcards_str'])
                basic_list = [-1] * (33 - len(df.at[rowid, 'cards_init']))
                card_list = list(df.at[rowid, 'cards_init'])
                card_list.extend(basic_list)
                f.write(os.linesep)
                # 将级牌拼接到牌组后面
                cards_str = ','.join([str(cards_id) for cards_id in card_list])
                current_rank = str(int(df.at[rowid, 'rank']))
                cards_str = f'{cards_str}||{current_rank}'
                f.write(cards_str)
                f.write(os.linesep)
                previous_card = df.at[rowid, 'leftcards_str']

    # 执行 CalculatorCardsValue.exe 获得拆牌信息
    subprocess.call(os.path.join(base_path, 'CalculatorCardsValue3.exe'))

    # 读取拆牌信息
    df_cardsvalue = pd.DataFrame(columns=['leftcards_str', 'cards_value', 'cards_type'])
    with open(os.path.join(base_path, 'Output.txt'), 'r') as fout:
        out = fout.readlines()
        for index, line in enumerate(out, start=1):
            if index % 5 == 1:
                df_cardsvalue.at[index - 1, 'leftcards_str'] = line.strip()
            if index % 5 == 3:
                df_cardsvalue.at[index - 3, 'cards_value'] = line.strip()
            if index % 5 == 0:
                df_cardsvalue.at[index - 5, 'cards_type'] = line.strip()

    df_cardsvalue.drop_duplicates(inplace=True)

    df = pd.merge(df, df_cardsvalue, on=['leftcards_str'], how='left', copy=False)
    df.drop(columns=["leftcards_str"], inplace=True)  # 删除合并标识列 leftcards_str

    # 存储拆牌后的数据
    write_data(df, filename='verify_cards_value_20190627')

    return df


if __name__ == '__main__':
    # reduce_data()
    first_init()
    cus, show, start = get_raw_data()
    # cus, show, start = get_test_data()
    df = judge_win_result(show, cus)  # 判断单双平
    df = pd.merge(df, start, on=["startguid", 'uid'])  # 合并胜率信息
    df.loc[:, 'win_ratio'] = df.apply(
        lambda row: round(row["win"] / (row["win"] + row["loss"]), 4) if (row["win"] + row["loss"]) != 0 else 0, axis=1)
    calculate_cards_value(df)  # 计算牌力值
