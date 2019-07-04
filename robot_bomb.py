# -*- encoding: utf-8 -*-
# @Version : 2.1
# @Time    : 2019/6/5 14:05
# @Time    : 2019/7/3 13:46
# @Author  : wanghd
# @note    : 用户牌局出炸情况处理（拆牌+ 统计）


import os
# import shutil
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import subprocess

# import profile
# import dask.dataframe as dd
# import win32api
# import win32con


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


def write_data(df, filedir=os.path.join(os.getcwd(), 'tmpdatadir1'), filename='robot_bomb', index=False):
    if not os.path.exists(filedir):
        os.mkdir(filedir)
    current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
    filename = f'{filename}_{current_time}.csv'
    df.to_csv(os.path.join(filedir, filename), index=index, header=True, encoding='gbk')


def get_raw_test_data(generate=False, startguid_list=None):
    """测试数据"""
    if generate:
        # 生成测试数据
        if startguid_list:
            show_files = [file for file in os.listdir(rawdatadir)
                          if file.startswith('short_win_ratio_0.5_showcards')][-1]
            cus_files = [file for file in os.listdir(rawdatadir)
                         if file.startswith('short_win_ratio_0.5_customization')][-1]
            if show_files and cus_files:
                show_file = pd.read_csv(os.path.join(rawdatadir, show_files), encoding='gbk')
                show_file = show_file.loc[show_file.loc[:, 'startguid'].isin(startguid_list)].reset_index(drop=True)
                show_file.to_csv(os.path.join(rawdatadir1, 'show_test.csv'), header=True, index=False, encoding='gbk')
                cus_file = pd.read_csv(os.path.join(rawdatadir, cus_files), encoding='gbk')
                cus_file = cus_file.loc[cus_file.loc[:, 'startguid'].isin(startguid_list)].reset_index(drop=True)
                cus_file.to_csv(os.path.join(rawdatadir1, 'cus_test.csv'), header=True, index=False, encoding='gbk')
            else:
                print("添加原始数据【缩减后】")
        else:
            print("添加startguid列表参数")
            sys.exit()
    else:
        # cus_usecols = ['startguid', 'uid', 'cards', 'num','rank'] 起手牌牌id[0-107], 起手牌牌数字[1-14]，级牌
        cus = pd.read_csv(os.path.join(rawdatadir1, 'cus_test.csv'), encoding='gbk')
        name_dict = {'cards': "cards_init", 'num': 'num_init'}
        cus.rename(columns=name_dict, inplace=True)
        # 对局id,uid, 开局时间，每手出牌时间点，出牌状态（出，不出，托管）'optype',
        # 牌型type，出牌回合, 出牌信息id[0-107]，出牌数字信息[1,14]
        # ['startguid', 'uid', 'starttime_unix', 'playtime_unix', 'type', 'cards_order', 'cards', 'num']
        show = pd.read_csv(os.path.join(rawdatadir1, 'show_test.csv'), encoding='gbk')
        cus['cards_order'] = 1
        mergedata = pd.merge(show, cus, on=['startguid', 'uid', 'cards_order'], how='left', copy=False)
        return mergedata


def get_raw_data(win_ratio=0.5):
    """读取正式数据"""
    # cus_usecols = ['startguid', 'uid', 'cards', 'num','rank'] 起手牌牌id[0-107], 起手牌牌数字[1-14]
    cus_usecols = ['startguid', 'uid', 'cards', 'rank', 'label_uid']  # 起手牌，级牌，胜率大于要求的用户uid标记
    cus_files = [file for file in os.listdir(rawdatadir) if
                 file.startswith(f'short_win_ratio_{win_ratio}_customization_20190625')]
    print(cus_files)
    name_dict = {'cards': "cards_init", 'num': 'num_init'}  # 重命名为 初始牌组

    # 对局id,uid, 开局时间，每手出牌时间点，出牌状态（出，不出，托管）'optype',
    # 牌型type，出牌回合, 出牌信息id[0-107]，出牌数字信息[1,14]
    # ['startguid', 'uid', 'starttime_unix', 'playtime_unix', 'type', 'cards_order', 'cards', 'num']
    show_usecols = ['startguid', 'uid', 'starttime_unix', 'playtime_unix', 'type', 'cards_order', 'cards']
    show_files = [file for file in os.listdir(rawdatadir) if
                  file.startswith(f'short_win_ratio_{win_ratio}_showcards_20190625')]
    print(show_files)
    for cus_file, show_file in zip(cus_files, show_files):
        cus = pd.read_csv(os.path.join(rawdatadir, cus_file), usecols=cus_usecols, encoding='gbk')  # 读取定制信息
        cus.rename(columns=name_dict, inplace=True)  # 重命名为初始牌组
        cus['cards_order'] = 1  # 起手牌标记出牌回合为 1
        show = pd.read_csv(os.path.join(rawdatadir, show_file), usecols=show_usecols, encoding='gbk')  # 读取出牌信息

        # 将初始牌组 和 出牌信息合并
        show = pd.merge(show, cus, on=['startguid', 'uid', 'cards_order'], how='left', copy=False)
        del cus
        return show


def reduce_raw_data(win_ratio=0.5):
    """缩减原始数据体积，仅保存需要的列， 用户胜率大于等于50%"""
    first_init()
    # 缩减customization 定制表体积
    cus_usecols = ['startguid', 'uid', 'cards', 'rank']  # 起手牌,级牌
    cus_files = [file for file in os.listdir(rawdatadir) if file.startswith('customization_20190625')]
    # 缩减showcards 出牌信息表体积
    show_usecols = ['startguid', 'uid', 'starttime_unix', 'playtime_unix', 'type', 'cards_order', 'cards']
    show_files = [file for file in os.listdir(rawdatadir) if file.startswith('showcards_20190625')]
    # 缩减start开局表，有历史对局战绩信息
    start_usecols = ['uid', 'win', 'loss']  # 历史胜局数，历史输局数
    start_files = [file for file in os.listdir(rawdatadir) if file.startswith('start_20190625')]

    for start_file, cus_file, show_file in zip(start_files, cus_files, show_files):
        start = pd.read_csv(os.path.join(rawdatadir, start_file), usecols=start_usecols, encoding='gbk')
        start.loc[:, "sum"] = start.apply(lambda row: row["win"] + row["loss"], axis=1)
        start.loc[:, "win_ratio"] = start.apply(lambda row: row["win"]/row["sum"] if row["sum"] != 0 else 0, axis=1)
        start = start.loc[start.loc[:, "win_ratio"] >= win_ratio].reset_index(drop=True)  # 筛选胜率>=50% 用户ID
        start.drop(columns=['loss'], inplace=True)  # 删除输局计数
        start.to_csv(os.path.join(rawdatadir, f"short_win_ratio_{win_ratio}_{start_file}"), index=False)  # 保存用户ID
        start.loc[:, 'label_uid'] = start.loc[:, 'win_ratio'].apply(lambda x: 1 if x >= win_ratio else 0)
        start.drop(columns=["win", 'sum', 'win_ratio'], inplace=True)  # 删除多余列
        start = start.groupby(["uid"]).agg({"label_uid": max}).reset_index(drop=False)  # 删除重复 uid 标记
        cus = pd.read_csv(os.path.join(rawdatadir, cus_file), usecols=cus_usecols, encoding='gbk')
        # cus = cus.loc[cus.loc[:, "uid"].isin(start_uid)].reset_index(drop=True)  # 不能筛减：对局不完整
        cus = pd.merge(cus, start, on='uid', how='left', copy=False)
        cus.loc[:, 'label_uid'] = cus.loc[:, 'label_uid'].fillna(0)
        cus.to_csv(os.path.join(rawdatadir, f"short_win_ratio_{win_ratio}_{cus_file}"), index=False, encoding='gbk')
        del cus
        show = pd.read_csv(os.path.join(rawdatadir, show_file), usecols=show_usecols, encoding='gbk')
        # show = show.loc[show.loc[:, "uid"].isin(start_uid)].reset_index(drop=True)  # 不能筛减
        # show = pd.merge(show, start, on='uid', how='left', copy=False) # cus和show一处标记label_uid
        # show.loc[:, 'label_uid'] = show.loc[:, 'label_uid'].fillna(0)  # cus和show一处标记label_uid
        show.to_csv(os.path.join(rawdatadir, f"short_win_ratio_{win_ratio}_{show_file}"), index=False, encoding='gbk')


def basic_treatment(mergedata):
    """
    对于原始局内信息做初步处理：级牌 、 构建累积出牌、 初始牌 、 剩余牌 、
    :param mergedata:
    :return:
    """
    # 填充对局的rank
    mergedata.loc[:, 'rank'] = mergedata.loc[:, 'rank'].fillna(method='ffill')
    # 填充胜率大于等于要求的uid标记
    mergedata.loc[:, 'label_uid'] = mergedata.loc[:, 'label_uid'].fillna(method='ffill')

    # write_data(mergedata, filename='mergedata')
    # # 将出牌ID 和 牌型ID 转为set
    # mergedata['cards'] =mergedata['cards'].apply(lambda x: set(str(x).split(',')) if x else pd.NaT)
    # mergedata['cards_init'] =mergedata['cards_init'].apply(lambda x: set(str(x).split(',')) if x else set())

    # 构建座位号
    seat_order = mergedata.loc[:, ['startguid', 'uid', 'playtime_unix']].sort_values(by=['startguid', 'playtime_unix'])
    seat_order = seat_order.groupby(["startguid"]).head(4)  # 仅取该局前4手牌
    seat_order.reset_index(drop=True, inplace=True)  # 重置index
    seat_order.loc[:, "seat_order"] = seat_order.index % 4 + 1
    # write_data(seat_order, filename='seat_order') #  输出中间结果
    seat_order.drop(columns=['playtime_unix'], inplace=True)
    mergedata = pd.merge(mergedata, seat_order, on=['startguid', 'uid'], copy=False)  # 合并座位号到原始df

    # 仅取出出牌的回合，用于构建 累积出牌信息
    mergedata_tmp = mergedata.loc[
        mergedata.loc[:, 'type'] > 0, ['startguid', 'uid', 'playtime_unix', 'cards']
    ].groupby(['startguid', 'uid']).head(100)

    # mergedata_tmp.loc[:, ['cards_init']] = mergedata_tmp.loc[:, ['cards_init']].fillna(method='ffill')
    mergedata_tmp = mergedata_tmp.sort_values(by=['startguid', 'uid', 'playtime_unix'])
    mergedata_tmp = mergedata_tmp.reset_index(drop=True)

    # mergedata_tmp.loc[:,'cards'] = mergedata_tmp.loc[:,'cards'].astype(str)
    mergedata_tmp.loc[:, 'cards'] = mergedata_tmp.loc[:, 'cards'] + ','
    # 获得累计出牌的ID 组
    mergedata_tmp_cum_cards = mergedata_tmp.groupby(['startguid', 'uid']).apply(lambda x: x.cards.cumsum())
    mergedata_tmp_cum_cards = mergedata_tmp_cum_cards.reset_index(drop=True)
    # 原cards列转化为累积 cum_cards列
    mergedata_tmp_cum_cards.rename(columns={"cards": "cum_cards"}, inplace=True)
    mergedata_tmp_cum_cards.drop(columns=['startguid', 'uid'])

    colnames = list(mergedata_tmp.columns)  # 构建新的列名 index
    colnames.append('cum_cards')  # 将累积出牌列 加入列名
    # write_data(mergedata_tmp,filename='mergedata_tmp')
    mergedata_tmp = pd.concat([mergedata_tmp, mergedata_tmp_cum_cards], axis=1, ignore_index=True)
    mergedata_tmp.columns = colnames  # 重置列名

    # 合并累积出牌信息到原始df
    mergedata_tmp.drop(columns=['cards'], inplace=True)
    mergedata = pd.merge(mergedata, mergedata_tmp, on=['startguid', 'uid', 'playtime_unix'], copy=False, how='left')
    # 填充未出牌时的 累积出牌信息
    mergedata = mergedata.sort_values(by=['startguid', 'uid', 'playtime_unix'])  # 按局、uid、出牌时间点排序
    mergedata = mergedata.reset_index(drop=True)  # 删除原index， 保证后续合并累积出牌不会错行
    mergedata_fill_cum_cards = mergedata.loc[:, ['startguid', 'uid', 'playtime_unix', 'cum_cards']].groupby(
        ['startguid', 'uid']).apply(lambda x: x.cum_cards.fillna(method='ffill'))
    mergedata_fill_cum_cards = mergedata_fill_cum_cards.reset_index(drop=True)
    # write_data(mergedata_fill_cum_cards, filename='mergedata_fill_cum_cards')

    mergedata.drop(columns=['cum_cards'], inplace=True)  # 删除仅包含出牌的累积出牌信息
    mergedata = pd.concat([mergedata, mergedata_fill_cum_cards], axis=1, )  # 将累积出牌合并到原df

    # 构建每一回合的剩余牌信息：ID组
    # 填充初始牌信息
    mergedata.loc[:, 'cards_init'] = mergedata.loc[:, 'cards_init'].fillna(method='ffill')
    # 整理累积出牌信息
    mergedata.loc[:, 'cum_cards'] = mergedata.loc[:, 'cum_cards'].astype(str)
    mergedata.loc[:, 'cum_cards'] = mergedata.loc[:, 'cum_cards'].apply(lambda x: set(x[:-1].split(sep=',')))
    # 整理初始牌信息
    mergedata.loc[:, 'cards_init'] = mergedata.loc[:, 'cards_init'].apply(lambda x: set(x.split(sep=',')))

    # 计算剩余牌信息 ID组
    mergedata.loc[:, 'leftcards'] = mergedata.apply(lambda row: row['cards_init'] - row['cum_cards'], axis=1)
    # 剩余牌数量
    mergedata.loc[:, 'leftcards_nums'] = mergedata.loc[:, 'leftcards'].apply(lambda x: len(x))
    mergedata.loc[:, 'leftcards_str'] = mergedata.loc[:, 'leftcards'].apply(lambda x: "".join(sorted(list(x))))
    mergedata.drop(columns=['cum_cards'], inplace=True)  # 删除累积出牌信息

    # 标记局第一手
    mergedata.loc[:, 'first_hand'] = mergedata.apply(
        lambda row: 1 if row['seat_order'] == 1 and row['cards_order'] == 1 else 0, axis=1)

    # write_data(mergedata, filename='robot')

    return mergedata


def calculate_lead_cards_value(df):
    """调用拆牌程序取 出牌 的牌值"""
    base_path = "F:/CardsValue"
    input_file = os.path.join(base_path, 'Input.txt')
    if os.path.exists(input_file):
        os.remove(input_file)
    output_file = os.path.join(base_path, 'Output.txt')
    if os.path.exists(output_file):
        os.remove(output_file)

    df_tmp = df.loc[df.loc[:, 'type'] > 0, ['startguid', 'uid', 'cards_order', 'rank', 'cards']]
    df_tmp = df_tmp.reset_index(drop=True)
    df_tmp.loc[:, 'leadcards_str'] = df_tmp.apply(
        lambda row: str(row["startguid"]) + str(row["uid"]) + str(row["cards_order"]), axis=1)
    # 写入牌面信息 到 input
    with open(input_file, 'a') as f:
        for rowid in range(df_tmp.shape[0]):
            f.write(df_tmp.at[rowid, 'leadcards_str'])
            card_list = str(df_tmp.at[rowid, 'cards']).split(',')
            basic_list = ['-1'] * (33 - len(card_list))
            card_list.extend(basic_list)
            f.write(os.linesep)
            # 将级牌拼接到牌组后面
            cards_str = ','.join([str(cards_id) for cards_id in card_list])
            try:
                current_rank = str(int(df.at[rowid, 'rank']))
            except ValueError:
                current_rank = '1'
            cards_str = f'{cards_str}||{current_rank}'
            f.write(cards_str)
            f.write(os.linesep)

    # del df_tmp  # 删除仅包含出牌的df_tmp
    # 执行 CalculatorCardsValue.exe 获得拆牌信息
    subprocess.call(os.path.join(base_path, 'CalculatorCardsValue3.exe'))

    # 读取拆牌信息
    df_cardsvalue = pd.DataFrame(columns=['leadcards_str', 'leadcards_cardvalue'])
    with open(output_file, 'r') as fout:
        out = fout.readlines()
        for index, line in enumerate(out, start=1):
            if index % 5 == 1:
                df_cardsvalue.at[index - 1, 'leadcards_str'] = line.strip()
            if index % 5 == 0:
                # 先按|| 牌型拆分拿到各个牌型组，在牌型组内 "-" 拆分拿到 牌值
                df_cardsvalue.at[index - 5, 'leadcards_cardvalue'] = line.strip().split("||")[-1].split('-')[-1]
            # print(index, line)

    df_cardsvalue.drop_duplicates(inplace=True)

    # ['startguid', 'uid', 'cards_order', 'rank', 'cards','leadcards_str','leadcards_cardvalue']
    df_tmp = pd.merge(df_tmp, df_cardsvalue, on=['leadcards_str'], how='left', copy=False)  # 合并牌值结果
    df_tmp.drop(columns=['rank', 'cards', 'leadcards_str'], inplace=True)  # 删除多余辅助列
    df = pd.merge(df, df_tmp, on=['startguid', 'uid', 'cards_order'], how="left", copy=False)  # 合并牌值结果

    return df


def calculate_cards_value(df):
    """调用拆牌程序拆解剩余手牌"""
    base_path = "F:/CardsValue"
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
            if df.at[rowid, 'leftcards_str'] and df.at[rowid, 'leftcards_str'] != previous_card:
                # 排除出完牌以及不出牌的情况
                f.write(df.at[rowid, 'leftcards_str'])
                basic_list = [-1] * (33 - len(df.at[rowid, 'leftcards']))
                card_list = list(df.at[rowid, 'leftcards'])
                card_list.extend(basic_list)
                f.write(os.linesep)
                # 将级牌拼接到牌组后面
                cards_str = ','.join([str(cards_id) for cards_id in card_list])
                try:
                    current_rank = str(int(df.at[rowid, 'rank']))
                except ValueError:
                    current_rank = '1'
                cards_str = f'{cards_str}||{current_rank}'
                f.write(cards_str)
                f.write(os.linesep)
                previous_card = df.at[rowid, 'leftcards_str']

    # 执行 CalculatorCardsValue.exe 获得拆牌信息
    subprocess.call(os.path.join(base_path, 'CalculatorCardsValue3.exe'))

    # 读取拆牌信息
    df_cardsvalue_cols = ['leftcards_str', 'cards_value', 'cards_type']
    # 加速读取速度
    df_cardsvalue = pd.read_csv(output_file, header=None, names=["result_abd"])
    df_cardsvalue.dropna(inplace=True)
    if df_cardsvalue.shape[0] % 4 == 0:
        df_cardsvalue.loc[:, "mark"] = df_cardsvalue.index // 4  # 验证结果数目
        df_cardsvalue = df_cardsvalue.groupby('mark').apply(lambda x: x.T)  # 转置分组后的 series，mark列会变成行
        df_cardsvalue = df_cardsvalue.loc[(slice(None), "result_abd"), :].reset_index(drop=True)  # 筛出结果，去除mark行
        df_cardsvalue.drop(columns=list(df_cardsvalue.columns)[-2], inplace=True)  # 删除结果中ID组列
        df_cardsvalue.columns = ['leftcards_str', 'cards_value', 'cards_type']
    else:
        df_cardsvalue = pd.DataFrame(columns=df_cardsvalue_cols)
        # 不满足结果数目的情况，使用 readline 保证完整记录可被记录
        with open(output_file, 'r') as fout:
            out = fout.readlines()
            for index, line in enumerate(out, start=1):
                if index % 5 == 1:
                    df_cardsvalue.at[index - 1, 'leftcards_str'] = line.strip()
                if index % 5 == 3:
                    df_cardsvalue.at[index - 3, 'cards_value'] = line.strip()
                if index % 5 == 0:
                    df_cardsvalue.at[index - 5, 'cards_type'] = line.strip()
                # print(index, line)

    df_cardsvalue.drop_duplicates(inplace=True)
    # 存储拆牌前的数据
    # write_data(df, filename='robot_001_notchaipai')

    df = pd.merge(df, df_cardsvalue, on=['leftcards_str'], how='left', copy=False)  # 合并牌力值拆牌信息
    df.drop(columns=["leftcards_str"], inplace=True)  # 删除合并标识列 leftcards_str

    df = calculate_lead_cards_value(df)  # 出牌牌值计算
    df = apart_cards_type(df)  # 牌型拆解
    df = rival_leadcards_treatment(df)  # 提取对手出牌，各出牌情况标记，对手剩余牌

    # 存储拆牌后的数据
    write_data(df, filedir=os.path.join(tmpdatadir1, '20190625'), filename='robot_result')

    return df


def apart_cards_type(df):
    """
    牌型拆解,
    :param df: 经过牌力值程序计算之后带有 牌型 的 dataframe
    :return: 各牌型数量，剩余手牌中各牌型的最大牌值，
    """

    cardoptypelist = [1, 2, 4, 64, 256, 512, 4096, 16384, 32768, 524288]

    def apart_card(x):
        """拆分牌组，取各牌型的数量 + 最大牌值"""
        # 1,2,4,64,256,512,4096,16384,32768,524288
        # 单，对，三，顺，连对，钢板，炸45，同花，超炸6-10,4王
        # cardoptypelist = [1, 2, 4, 64, 256, 512, 4096, 16384, 32768, 524288]
        cards_list = str(x).split(sep='||')
        if 'nan' not in cards_list:
            card_optype_list = [card.split(sep='-')[0] for card in cards_list]  # 牌类型
            # card_nums_list = [card.split(sep='-')[1] for card in cards_list]  # 牌数量
            card_value_list = [card.split(sep='-')[2] for card in cards_list]  # 牌值
            # 取各牌型数量
            optype_counter = Counter(card_optype_list)
            # 各牌型数量列名—— 用于之后分列 作为列名
            # colnames_optypenumlist = [f"optype{optype}_nums" for optype in cardoptypelist]
            # 各牌型 手牌中的最大牌值，没有牌型则为0
            # colnames_optypemaxcardvaluelist = [f"optype{optype}_maxcardvalue" for optype in cardoptypelist]
            # 各牌型数量 int 数值
            optypenumlist = [optype_counter[str(optype_nums)] for optype_nums in cardoptypelist]
            optype_maxcardvaluelist = []
            for index, optypenum in enumerate(optypenumlist, start=1):
                if optypenum > 0:
                    optype_maxcardvaluelist.append(card_value_list[sum(optypenumlist[:index]) - 1])
                else:
                    optype_maxcardvaluelist.append(0)  # 不存在该牌型则记0

            optypenumlist.extend(optype_maxcardvaluelist)  # 各牌型数量+ 各牌型最大牌值
            return ",".join([str(x) for x in optypenumlist])
        else:
            return ",".join(list('0' * 20))

    # 手牌内各牌型的数量，各牌型最大牌值
    df.loc[:, 'apartcol'] = df.loc[:, 'cards_type'].apply(apart_card)
    # 得到各牌型之后进行列拆分
    colnames_optypenumlist = [f"optype{optype}_nums" for optype in cardoptypelist]
    colnames_optypemaxcardvaluelist = [f"optype{optype}_maxcardvalue" for optype in cardoptypelist]
    colnames_optypenumlist.extend(colnames_optypemaxcardvaluelist)
    # print(colnames_optypenumlist)
    # 将拆分得到的牌型数量，牌型内最大牌值 展开到 列
    df_apart = df.loc[:, 'apartcol'].str.split(',', expand=True)
    # 重命名展开的列名
    df_apart.columns = colnames_optypenumlist
    for colname in colnames_optypenumlist:
        df_apart.loc[:, colname] = df_apart.loc[:, colname].astype(int)  # 修改所有拆分列为int
    # 将展开列 合并回原始 df ,并删除拆分列
    df = pd.concat([df, df_apart], axis=1)
    df.drop(columns=['apartcol'], inplace=True)

    # 统计剩余手牌的信息
    #  1.剩余最大炸类型，2.剩余最大炸 牌数，3.剩余炸弹数,4 非炸弹手数：出完牌的手数,5 非炸牌型平均牌值
    result_cols = ["leftmaxbomb", 'leftmaxbomb_cardnum', 'leftbomb_nums', 'Nonebomb_lefthands',
                   'Nonebomb_MeanCardvalue']

    def calculate_lefthand_cards(x):
        """判断剩余最大炸弹的大小，【4炸小于10 --1，同花顺--2，其他--0】
        剩余最大炸的牌的数量
        剩余炸弹数
        非炸弹牌型手数
        非炸弹牌型平均牌值"""
        result_list = []
        bomb_list = []  # 由炸弹 (cardtype, cardnum, cardvalue) 元组构成的列表
        not_bomb_list = []  # 非炸手牌的牌型列表
        not_bomb_mean_cards_value = []  # 非炸手牌牌型牌值列表
        cards_list = str(x).split(sep='||')  # ['1-1-8','2-2-10'],
        for card in cards_list:
            if 'nan' not in card:
                card_type, card_num, card_value = card.split(sep='-')
                if card_type in ['4096', '16384', '32768', '524288']:
                    bomb_list.append((card_type, card_num, card_value))
                else:
                    not_bomb_list.append(card_type)  # 非炸手牌牌型值
                    not_bomb_mean_cards_value.append(int(card_value))  # 非炸手牌牌值

        bomb_list_length = len(bomb_list)  # 剩余炸弹数量
        if bomb_list_length > 0:
            card_type, card_num, card_value = bomb_list[-1]
            # 最大炸类型
            if card_type in ['16384', '32768', '524288']:
                result_list.append(2)
            elif card_type in ['4096'] and card_num in ['4'] and int(card_value) < 9:
                result_list.append(1)
            else:
                result_list.append(0)
            result_list.append(int(card_num))  # 最大炸牌数
        else:
            result_list.append(0)  # 最大炸类型
            result_list.append(0)  # 最大炸牌数
        result_list.append(bomb_list_length)  # 剩余炸弹数

        # 非炸其他所有牌型数量 - min{对子数，三张数}
        shoushu = len(not_bomb_list) - min(not_bomb_list.count('2'), not_bomb_list.count("4"))
        result_list.append(shoushu)

        # 非炸手牌平均牌值
        result_list.append(np.round(np.mean(not_bomb_mean_cards_value), 2))
        return ",".join([str(x) for x in result_list])

    # 判断剩余牌统计值
    df.loc[:, 'apartcol'] = df.loc[:, 'cards_type'].apply(calculate_lefthand_cards)

    # 将拆分得到的剩余牌信息 展开到 列
    df_apart = df.loc[:, 'apartcol'].str.split(',', expand=True)
    # 重命名展开的列名
    df_apart.columns = result_cols
    for colname in result_cols[:-1]:
        df_apart.loc[:, colname] = df_apart.loc[:, colname].astype(int)  # 修改所有拆分列为int
    # 将展开列 合并回原始 df ,并删除拆分列
    df = pd.concat([df, df_apart], axis=1)
    df.drop(columns=['apartcol'], inplace=True)

    return df


def get_unique_max(sequence):
    """取非重复值中的最大值： 非重复值指只出现一次的值，如果出现多次，删除该值"""
    sequence = sequence.astype(str)
    counter = Counter(sequence).items()
    maxvalue = max([int(x[0]) if x[1] == 1 else -1 for x in counter])
    return maxvalue


def rival_leadcards_treatment(df):
    """提取需要的统计数据的基础数据"""
    # 循环处理每一局
    df.sort_values(by=["startguid", 'playtime_unix'], ascending=[True, True])
    df = df.reset_index(drop=True)
    start_guids = df.loc[:, 'startguid'].unique()
    statistic_df = pd.DataFrame()

    def compare_bomb(compare_df, compare_idx, plus_idx):
        bomb_type_list = [4096, 16384, 32768, 524288]
        bomb_array = compare_df.loc[compare_idx + plus_idx, ['optype4096_maxcardvalue',
                                                             'optype16384_maxcardvalue',
                                                             'optype32768_maxcardvalue',
                                                             'optype524288_maxcardvalue']].values
        inhand_bomb_max_cardvalue = int(np.max(bomb_array))  # 最大炸牌值
        inhand_bomb_max_type = bomb_type_list[int(np.argmax(bomb_array))]  # 最大炸类型
        # 判断炸弹类型大小
        # 比类型
        if inhand_bomb_max_type > compare_df.at[compare_idx, 'type']:
            compare_df.at[compare_idx + plus_idx, 'lead_bomb'] = 1  # 标记对手出炸
            compare_df.at[compare_idx + plus_idx, 'need_bomb'] = 1  # 需要出炸
            if compare_df.at[compare_idx, "first_hand"] > 0:
                compare_df.at[compare_idx + plus_idx, 'lead_firsthand_bomb'] = 1  # 标记为开局出炸
            if compare_df.at[compare_idx + plus_idx, 'type'] > 0:
                compare_df.at[compare_idx + plus_idx, 'label_bomb'] = 1  # 出炸
        # 类型相同比牌数
        elif inhand_bomb_max_type == compare_df.at[compare_idx, 'type']:
            bomb_card_nums = len(str(compare_df.at[compare_idx, 'cards']).split(","))  # 出牌炸弹的牌数
            if compare_df.at[compare_idx + plus_idx, 'leftmaxbomb_cardnum'] > bomb_card_nums:
                compare_df.at[compare_idx + plus_idx, 'lead_bomb'] = 1  # 标记对手出炸
                compare_df.at[compare_idx + plus_idx, 'need_bomb'] = 1  # 需要出炸
                if compare_df.at[compare_idx, "first_hand"] > 0:
                    compare_df.at[compare_idx + plus_idx, 'lead_firsthand_bomb'] = 1  # 标记为开局出炸
                if compare_df.at[compare_idx + plus_idx, 'type'] > 0:
                    compare_df.at[compare_idx + plus_idx, 'label_bomb'] = 1  # 出炸
            # 牌数相同比牌值
            elif compare_df.at[compare_idx + plus_idx, 'leftmaxbomb_cardnum'] == bomb_card_nums:
                if inhand_bomb_max_cardvalue > int(compare_df.at[compare_idx, 'leadcards_cardvalue']):
                    compare_df.at[compare_idx + plus_idx, 'lead_bomb'] = 1  # 标记对手出炸
                    compare_df.at[compare_idx + plus_idx, 'need_bomb'] = 1  # 需要出炸
                    if compare_df.at[compare_idx, "first_hand"] > 0:
                        compare_df.at[compare_idx + plus_idx, 'lead_firsthand_bomb'] = 1  # 标记为开局出炸
                    if compare_df.at[compare_idx + plus_idx, 'type'] > 0:
                        compare_df.at[compare_idx + plus_idx, 'label_bomb'] = 1  # 出炸
        return compare_df

    for start_guid in start_guids:
        gamedf = df.loc[df.loc[:, 'startguid'] == start_guid]
        gamedf = gamedf.sort_values(by=["playtime_unix"], ascending=True)  # 根据出牌顺序排序
        gamedf.reset_index(drop=True, inplace=True)
        gamedf.loc[:, 'lead_firsthand_bomb'] = 0  # 局第一手炸弹
        gamedf.loc[:, 'lead_bomb'] = 0  # 出炸
        gamedf.loc[:, 'lead_max_optype1'] = 0  # 最大单张
        gamedf.loc[:, 'lead_max_optype2'] = 0  # 最大对子
        gamedf.loc[:, 'lead_max_optype4'] = 0  # 最大三张
        gamedf.loc[:, 'lead_max_optype32'] = 0  # 最大三带二
        gamedf.loc[:, 'lead_max_optype256'] = 0  # 连对
        gamedf.loc[:, 'lead_max_optype512'] = 0  # 钢板
        gamedf.loc[:, 'lead_max_optype64'] = 0  # 顺子
        gamedf.loc[:, 'optype12432'] = 0  # 单张，对子，三张，三带二压不住
        gamedf.loc[:, 'optype25651264'] = 0  # 连对，钢板，顺子压不住
        gamedf.loc[:, 'need_bomb'] = 2  # 是否需要出炸 0非1是2其他情况
        gamedf.loc[:, 'label_bomb'] = 0  # 是否出炸
        # 取队友剩余牌
        gamedf.loc[:, "leftcards_nums_pair"] = gamedf.loc[:, 'leftcards_nums'].shift(-2).shift(4)
        gamedf.at[0, "leftcards_nums_pair"] = 27
        gamedf.at[1, "leftcards_nums_pair"] = 27
        gamedf.at[2, "leftcards_nums_pair"] = gamedf.at[0, 'leftcards_nums']
        gamedf.at[3, "leftcards_nums_pair"] = gamedf.at[1, 'leftcards_nums']

        idx_length = gamedf.shape[0]
        gamedf.loc[:, 'type'] = gamedf.loc[:, 'type'].astype(int)
        # 取出牌对手剩余手牌数量，对手的队友的剩余牌数量
        rival_df = gamedf.loc[:, ['playtime_unix', 'leftcards_nums', 'type']]
        rival_df.loc[:, 'rival_leftcards_nums'] = 28  # 出牌对手的剩余牌数量 , 28表示无需记录
        rival_df.loc[:, 'rival_leftcards_nums_pair'] = 28  # 出牌对手队友的剩余牌数量， 28表示无需记录
        # 局第一手
        rival_df.at[1, 'rival_leftcards_nums'] = rival_df.at[0, 'leftcards_nums']
        rival_df.at[1, 'rival_leftcards_nums_pair'] = 27
        if rival_df.at[1, 'type'] > 0:
            # 局第两手的情况
            rival_df.at[2, 'rival_leftcards_nums'] = rival_df.at[1, 'leftcards_nums']
            rival_df.at[2, 'rival_leftcards_nums_pair'] = 27
            if rival_df.at[2, 'type'] == 0 and rival_df.at[3, 'type'] == 0:
                # 第二位， 不出，不出 的情况
                rival_df.at[4, 'rival_leftcards_nums'] = rival_df.at[1, 'leftcards_nums']
                rival_df.at[4, 'rival_leftcards_nums_pair'] = 27
        for idx in range(2, idx_length-3):
            if rival_df.at[idx, 'type'] > 0:
                rival_df.at[idx + 1, 'rival_leftcards_nums'] = rival_df.at[idx, 'leftcards_nums']  # 出牌对手
                rival_df.at[idx + 1, 'rival_leftcards_nums_pair'] = rival_df.at[idx - 2, 'leftcards_nums']  # 出牌对手队友
                if rival_df.at[idx + 1, 'type'] == 0 and rival_df.at[idx + 2, 'type'] == 0:
                    # 第二位， 不出，不出 的情况
                    rival_df.at[idx + 3, 'rival_leftcards_nums'] = rival_df.at[idx, 'leftcards_nums']  # 出牌对手
                    rival_df.at[idx+3, 'rival_leftcards_nums_pair'] = rival_df.at[idx+2, 'leftcards_nums']
        for idx in [idx_length-3, idx_length-2]:
            # 局倒数第三手，第二手的情况
            if rival_df.at[idx, 'type'] > 0:
                rival_df.at[idx+1, 'rival_leftcards_nums'] = rival_df.at[idx, 'leftcards_nums']
                rival_df.at[idx+1, 'rival_leftcards_nums_pair'] = rival_df.at[idx-2, 'leftcards_nums']  # 出牌对手队友
        rival_df.drop(columns=["leftcards_nums", 'type'], inplace=True)
        gamedf = pd.merge(gamedf, rival_df, on='playtime_unix', copy=False)  # 根据出牌时间来匹配剩余牌数

        for idx in range(idx_length - 3):
            if gamedf.at[idx, 'type'] in [4096, 16384, 32768, 524288]:
                # 对手出炸弹
                gamedf = compare_bomb(gamedf, idx, 1)
                if gamedf.at[idx+1, 'type'] == 0 and gamedf.at[idx+2, 'type'] == 0:
                    # 第二位，不出，不出 的情况
                    gamedf = compare_bomb(gamedf, idx, 3)

            elif gamedf.at[idx, 'type'] in [1]:
                # 出单张
                # 判断是否最大单张
                max_optype1_cardvalue = get_unique_max(gamedf.loc[idx+1:idx+3, 'optype1_maxcardvalue'])
                if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 1, 'optype1_maxcardvalue']):
                    # 单牌压不住的情况, 包含出牌为最大单张的情况
                    if gamedf.at[idx + 1, 'leftbomb_nums'] > 0:
                        gamedf.at[idx + 1, 'need_bomb'] = 1
                        gamedf.at[idx + 1, 'optype12432'] = 1  # 单张，对子，三张，三带二压不住
                        if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(max_optype1_cardvalue):
                            # 为最大单张的标记
                            gamedf.loc[idx + 1, 'lead_max_optype1'] = 1
                        if gamedf.at[idx + 1, 'type'] > 0:
                            gamedf.at[idx + 1, 'label_bomb'] = 1  # 出炸
                if gamedf.at[idx+1, 'type'] == 0 and gamedf.at[idx+2, 'type'] == 0:
                    # 第二位队友的情况（不出，不出），压不住包含出牌为最大单张的情况
                    if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 3, 'optype1_maxcardvalue']):
                        # 单牌压不住的情况
                        if gamedf.at[idx + 3, 'leftbomb_nums'] > 0:
                            gamedf.at[idx + 3, 'need_bomb'] = 1
                            gamedf.at[idx + 3, 'optype12432'] = 1  # 单张，对子，三张，三带二压不住
                            if int(gamedf.at[idx, 'leadcards_cardvalue']) == int(max_optype1_cardvalue):
                                # 为最大单张的标记
                                gamedf.loc[idx + 3, 'lead_max_optype1'] = 1
                            if gamedf.at[idx + 3, 'type'] > 0:
                                gamedf.at[idx + 3, 'label_bomb'] = 1  # 出炸
            elif gamedf.at[idx, 'type'] in [2]:
                # 出对子
                # 判断是否最大对子
                max_optype1_cardvalue = get_unique_max(gamedf.loc[idx+1:idx+3, 'optype2_maxcardvalue'])
                if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 1, 'optype2_maxcardvalue']):
                    # 对子压不住的情况, 包含出牌为最大对子的情况
                    if gamedf.at[idx + 1, 'leftbomb_nums'] > 0:
                        gamedf.at[idx + 1, 'need_bomb'] = 1
                        gamedf.at[idx + 1, 'optype12432'] = 1  # 单张，对子，三张，三带二压不住
                        if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(max_optype1_cardvalue):
                            # 为最大对子的标记
                            gamedf.loc[idx + 1, 'lead_max_optype2'] = 1
                        if gamedf.at[idx + 1, 'type'] > 0:
                            gamedf.at[idx + 1, 'label_bomb'] = 1  # 出炸
                if gamedf.at[idx+1, 'type'] == 0 and gamedf.at[idx+2, 'type'] == 0:
                    # 第二位队友的情况（不出，不出），压不住包含出牌为最大单张的情况
                    if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 3, 'optype2_maxcardvalue']):
                        # 对子压不住的情况
                        if gamedf.at[idx + 3, 'leftbomb_nums'] > 0:
                            gamedf.at[idx + 3, 'need_bomb'] = 1
                            gamedf.at[idx + 3, 'optype12432'] = 1  # 单张，对子，三张，三带二压不住
                            if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(max_optype1_cardvalue):
                                # 为最大对子的标记
                                gamedf.loc[idx + 3, 'lead_max_optype2'] = 1
                            if gamedf.at[idx + 3, 'type'] > 0:
                                gamedf.at[idx + 3, 'label_bomb'] = 1  # 出炸
            elif gamedf.at[idx, 'type'] in [4]:
                # 出三张
                # 判断是否最大三张
                max_optype1_cardvalue = get_unique_max(gamedf.loc[idx+1:idx+3, 'optype4_maxcardvalue'])
                if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 1, 'optype4_maxcardvalue']):
                    # 三张压不住的情况, 包含出牌为最大三张的情况
                    if gamedf.at[idx + 1, 'leftbomb_nums'] > 0:
                        gamedf.at[idx + 1, 'need_bomb'] = 1
                        gamedf.at[idx + 1, 'optype12432'] = 1  # 单张，对子，三张，三带二压不住
                        if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(max_optype1_cardvalue):
                            # 为最大三张的标记
                            gamedf.loc[idx + 1, 'lead_max_optype4'] = 1
                        if gamedf.at[idx + 1, 'type'] > 0:
                            gamedf.at[idx + 1, 'label_bomb'] = 1  # 出炸
                if gamedf.at[idx+1, 'type'] == 0 and gamedf.at[idx+2, 'type'] == 0:
                    # 第二位队友的情况（不出，不出），压不住包含出牌为最大三张的情况
                    if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 3, 'optype4_maxcardvalue']):
                        # 三张压不住的情况
                        if gamedf.at[idx + 3, 'leftbomb_nums'] > 0:
                            gamedf.at[idx + 3, 'need_bomb'] = 1
                            gamedf.at[idx + 3, 'optype12432'] = 1  # 单张，对子，三张，三带二压不住
                            if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(max_optype1_cardvalue):
                                # 为最大三张的标记
                                gamedf.loc[idx + 3, 'lead_max_optype4'] = 1
                            if gamedf.at[idx + 3, 'type'] > 0:
                                gamedf.at[idx + 3, 'label_bomb'] = 1  # 出炸
            elif gamedf.at[idx, 'type'] in [32]:
                # 出三带二，包含最大三带二的情况
                # 判断三张是不是最大的
                max_optype1_cardvalue = get_unique_max(gamedf.loc[idx + 1:idx + 3, 'optype4_maxcardvalue'])
                if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 1, 'optype4_maxcardvalue']):
                    # 三张压不住的情况, 包含出牌为最大三张的情况
                    if gamedf.at[idx + 1, 'leftbomb_nums'] > 0:
                        gamedf.at[idx + 1, 'need_bomb'] = 1
                        gamedf.at[idx + 1, 'optype12432'] = 1  # 单张，对子，三张，三带二压不住
                        if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(max_optype1_cardvalue):
                            # 为最大三带二的标记
                            gamedf.loc[idx + 1, 'lead_max_optype32'] = 1
                        if gamedf.at[idx + 1, 'type'] > 0:
                            gamedf.at[idx + 1, 'label_bomb'] = 1  # 出炸
                else:
                    # 三张压的住，但是没有对子
                    if int(gamedf.at[idx+1, 'optype2_nums']) < 1 and gamedf.at[idx + 1, 'leftbomb_nums'] > 0:
                        gamedf.at[idx + 1, 'need_bomb'] = 1
                        gamedf.loc[idx + 1, 'lead_max_optype32'] = 1
                        gamedf.at[idx + 1, 'optype12432'] = 1  # 单张，对子，三张，三带二压不住
                        if gamedf.at[idx + 1, 'type'] > 0:
                            gamedf.at[idx + 1, 'label_bomb'] = 1  # 出炸
                if gamedf.at[idx + 1, 'type'] == 0 and gamedf.at[idx + 2, 'type'] == 0:
                    # 第二位队友的情况（不出，不出），压不住包含出牌为最大三张的情况
                    if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 3, 'optype4_maxcardvalue']):
                        # 三张压不住的情况
                        if gamedf.at[idx + 3, 'leftbomb_nums'] > 0:
                            gamedf.at[idx + 3, 'need_bomb'] = 1
                            gamedf.at[idx + 3, 'optype12432'] = 1  # 单张，对子，三张，三带二压不住
                            if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(max_optype1_cardvalue):
                                # 为最大三张的标记
                                gamedf.loc[idx + 3, 'lead_max_optype32'] = 1
                            if gamedf.at[idx + 3, 'type'] > 0:
                                gamedf.at[idx + 3, 'label_bomb'] = 1  # 出炸
                else:
                    # 三张压的住，但是没有对子
                    if int(gamedf.at[idx+3, 'optype2_nums']) < 1 and gamedf.at[idx + 3, 'leftbomb_nums'] > 0:
                        gamedf.at[idx + 3, 'need_bomb'] = 1
                        gamedf.loc[idx + 3, 'lead_max_optype32'] = 1
                        gamedf.at[idx + 3, 'optype12432'] = 1  # 单张，对子，三张，三带二压不住
                        if gamedf.at[idx + 3, 'type'] > 0:
                            gamedf.at[idx + 3, 'label_bomb'] = 1  # 出炸
            # 连对，钢板，顺子
            elif gamedf.at[idx, 'type'] in [256]:
                # 连对
                if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 1, 'optype256_maxcardvalue']):
                    # 连对压不住的情况
                    if gamedf.at[idx + 1, 'leftbomb_nums'] > 0:
                        gamedf.at[idx + 1, 'need_bomb'] = 1
                        gamedf.at[idx + 1, 'lead_max_optype256'] = 1
                        gamedf.at[idx + 1, 'optype25651264'] = 1  # 连对，钢板，顺子压不住
                        if gamedf.at[idx + 1, 'type'] > 0:
                            gamedf.at[idx + 1, 'label_bomb'] = 1  # 出炸
                if gamedf.at[idx+1, 'type'] == 0 and gamedf.at[idx+2, 'type'] == 0:
                    # 第二位队友的情况（不出，不出），
                    if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 3, 'optype256_maxcardvalue']):
                        # 连对压不住的情况
                        if gamedf.at[idx + 3, 'leftbomb_nums'] > 0:
                            gamedf.at[idx + 3, 'need_bomb'] = 1
                            gamedf.at[idx + 3, 'lead_max_optype256'] = 1
                            gamedf.at[idx + 3, 'optype25651264'] = 1  # 连对，钢板，顺子压不住
                            if gamedf.at[idx + 3, 'type'] > 0:
                                gamedf.at[idx + 3, 'label_bomb'] = 1  # 出炸
            elif gamedf.at[idx, 'type'] in [512]:
                # 钢板
                if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 1, 'optype512_maxcardvalue']):
                    # 钢板压不住的情况
                    if gamedf.at[idx + 1, 'leftbomb_nums'] > 0:
                        gamedf.at[idx + 1, 'need_bomb'] = 1
                        gamedf.at[idx + 1, 'lead_max_optype512'] = 1
                        gamedf.at[idx + 1, 'optype25651264'] = 1  # 连对，钢板，顺子压不住
                        if gamedf.at[idx + 1, 'type'] > 0:
                            gamedf.at[idx + 1, 'label_bomb'] = 1  # 出炸
                if gamedf.at[idx+1, 'type'] == 0 and gamedf.at[idx+2, 'type'] == 0:
                    # 第二位队友的情况（不出，不出），
                    if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 3, 'optype512_maxcardvalue']):
                        # 钢板压不住的情况
                        if gamedf.at[idx + 3, 'leftbomb_nums'] > 0:
                            gamedf.at[idx + 3, 'need_bomb'] = 1
                            gamedf.at[idx + 3, 'lead_max_optype512'] = 1
                            gamedf.at[idx + 3, 'optype25651264'] = 1  # 连对，钢板，顺子压不住
                            if gamedf.at[idx + 3, 'type'] > 0:
                                gamedf.at[idx + 3, 'label_bomb'] = 1  # 出炸
            elif gamedf.at[idx, 'type'] in [64]:
                # 顺子
                if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 1, 'optype64_maxcardvalue']):
                    # 顺子压不住的情况
                    if gamedf.at[idx + 1, 'leftbomb_nums'] > 0:
                        gamedf.at[idx + 1, 'need_bomb'] = 1
                        gamedf.at[idx + 1, 'lead_max_optype64'] = 1
                        gamedf.at[idx + 1, 'optype25651264'] = 1  # 连对，钢板，顺子压不住
                        if gamedf.at[idx + 1, 'type'] > 0:
                            gamedf.at[idx + 1, 'label_bomb'] = 1  # 出炸
                if gamedf.at[idx+1, 'type'] == 0 and gamedf.at[idx+2, 'type'] == 0:
                    # 第二位队友的情况（不出，不出），
                    if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 3, 'optype64_maxcardvalue']):
                        # 顺子压不住的情况
                        if gamedf.at[idx + 3, 'leftbomb_nums'] > 0:
                            gamedf.at[idx + 3, 'need_bomb'] = 1
                            gamedf.at[idx + 3, 'lead_max_optype64'] = 1
                            gamedf.at[idx + 3, 'optype25651264'] = 1  # 连对，钢板，顺子压不住
                            if gamedf.at[idx + 3, 'type'] > 0:
                                gamedf.at[idx + 3, 'label_bomb'] = 1  # 出炸

        for idx in range(idx_length-3, idx_length-1):
            # 倒数第三手和倒数第二手的情况
            if gamedf.at[idx, 'type'] in [4096, 16384, 32768, 524288]:
                # 对手出炸弹
                gamedf = compare_bomb(gamedf, idx, 1)
            elif gamedf.at[idx, 'type'] in [1]:
                # 出单张
                max_optype1_cardvalue = get_unique_max(gamedf.loc[idx+1:, 'optype1_maxcardvalue'])
                if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 1, 'optype1_maxcardvalue']):
                    # 单牌压不住的情况, 包含出牌为最大单张的情况
                    if gamedf.at[idx + 1, 'leftbomb_nums'] > 0:
                        gamedf.at[idx + 1, 'need_bomb'] = 1
                        gamedf.at[idx + 1, 'optype12432'] = 1  # 单张，对子，三张，三带二压不住
                        if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(max_optype1_cardvalue):
                            # 为最大单张的标记
                            gamedf.loc[idx + 1, 'lead_max_optype1'] = 1
                        if gamedf.at[idx + 1, 'type'] > 0:
                            gamedf.at[idx + 1, 'label_bomb'] = 1  # 出炸
            elif gamedf.at[idx, 'type'] in [2]:
                # 出对子
                max_optype1_cardvalue = get_unique_max(gamedf.loc[idx+1:, 'optype2_maxcardvalue'])
                if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 1, 'optype2_maxcardvalue']):
                    # 对子压不住的情况, 包含出牌为最大对子的情况
                    if gamedf.at[idx + 1, 'leftbomb_nums'] > 0:
                        gamedf.at[idx + 1, 'need_bomb'] = 1
                        gamedf.at[idx + 1, 'optype12432'] = 1  # 单张，对子，三张，三带二压不住
                        if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(max_optype1_cardvalue):
                            # 为最大对子的标记
                            gamedf.loc[idx + 1, 'lead_max_optype2'] = 1
                        if gamedf.at[idx + 1, 'type'] > 0:
                            gamedf.at[idx + 1, 'label_bomb'] = 1  # 出炸
            elif gamedf.at[idx, 'type'] in [4]:
                # 出三张
                max_optype1_cardvalue = get_unique_max(gamedf.loc[idx+1:, 'optype4_maxcardvalue'])
                if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 1, 'optype4_maxcardvalue']):
                    # 三张压不住的情况, 包含出牌为最大三张的情况
                    if gamedf.at[idx + 1, 'leftbomb_nums'] > 0:
                        gamedf.at[idx + 1, 'need_bomb'] = 1
                        gamedf.at[idx + 1, 'optype12432'] = 1  # 单张，对子，三张，三带二压不住
                        if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(max_optype1_cardvalue):
                            # 为最大三张的标记
                            gamedf.loc[idx + 1, 'lead_max_optype4'] = 1
                        if gamedf.at[idx + 1, 'type'] > 0:
                            gamedf.at[idx + 1, 'label_bomb'] = 1  # 出炸
            elif gamedf.at[idx, 'type'] in [32]:
                # 出三带二，包含最大三带二的情况
                # 判断三张是不是最大的
                max_optype1_cardvalue = get_unique_max(gamedf.loc[idx + 1:idx + 3, 'optype4_maxcardvalue'])
                if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 1, 'optype4_maxcardvalue']):
                    # 三张压不住的情况, 包含出牌为最大三张的情况
                    if gamedf.at[idx + 1, 'leftbomb_nums'] > 0:
                        gamedf.at[idx + 1, 'need_bomb'] = 1
                        gamedf.at[idx + 1, 'optype12432'] = 1  # 单张，对子，三张，三带二压不住
                        if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(max_optype1_cardvalue):
                            # 为最大三带二的标记
                            gamedf.loc[idx + 1, 'lead_max_optype32'] = 1
                        if gamedf.at[idx + 1, 'type'] > 0:
                            gamedf.at[idx + 1, 'label_bomb'] = 1  # 出炸
                else:
                    # 三张压的住，但是没有对子
                    if int(gamedf.at[idx+1, 'optype2_nums']) < 1 and gamedf.at[idx + 1, 'leftbomb_nums'] > 0:
                        gamedf.at[idx + 1, 'need_bomb'] = 1
                        gamedf.loc[idx + 1, 'lead_max_optype32'] = 1
                        gamedf.at[idx + 1, 'optype12432'] = 1  # 单张，对子，三张，三带二压不住
                        if gamedf.at[idx + 1, 'type'] > 0:
                            gamedf.at[idx + 1, 'label_bomb'] = 1  # 出炸
            elif gamedf.at[idx, 'type'] in [256]:
                # 连对
                if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 1, 'optype256_maxcardvalue']):
                    # 连对压不住的情况
                    if gamedf.at[idx + 1, 'leftbomb_nums'] > 0:
                        gamedf.at[idx + 1, 'need_bomb'] = 1
                        gamedf.at[idx + 1, 'lead_max_optype256'] = 1
                        gamedf.at[idx + 1, 'optype25651264'] = 1  # 连对，钢板，顺子压不住
                        if gamedf.at[idx + 1, 'type'] > 0:
                            gamedf.at[idx + 1, 'label_bomb'] = 1  # 出炸
            elif gamedf.at[idx, 'type'] in [512]:
                # 钢板
                if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 1, 'optype512_maxcardvalue']):
                    # 钢板压不住的情况
                    if gamedf.at[idx + 1, 'leftbomb_nums'] > 0:
                        gamedf.at[idx + 1, 'need_bomb'] = 1
                        gamedf.at[idx + 1, 'lead_max_optype512'] = 1
                        gamedf.at[idx + 1, 'optype25651264'] = 1  # 连对，钢板，顺子压不住
                        if gamedf.at[idx + 1, 'type'] > 0:
                            gamedf.at[idx + 1, 'label_bomb'] = 1  # 出炸
            elif gamedf.at[idx, 'type'] in [64]:
                # 顺子
                if int(gamedf.at[idx, 'leadcards_cardvalue']) >= int(gamedf.at[idx + 1, 'optype64_maxcardvalue']):
                    # 顺子压不住的情况
                    if gamedf.at[idx + 1, 'leftbomb_nums'] > 0:
                        gamedf.at[idx + 1, 'need_bomb'] = 1
                        gamedf.at[idx + 1, 'lead_max_optype64'] = 1
                        gamedf.at[idx + 1, 'optype25651264'] = 1  # 连对，钢板，顺子压不住
                        if gamedf.at[idx + 1, 'type'] > 0:
                            gamedf.at[idx + 1, 'label_bomb'] = 1  # 出炸

        gamedf_cols = ['playtime_unix', "leftcards_nums_pair", 'rival_leftcards_nums', 'rival_leftcards_nums_pair',
                       'need_bomb', 'label_bomb', 'lead_firsthand_bomb', 'lead_bomb', 'lead_max_optype1',
                       'lead_max_optype2', 'lead_max_optype4', 'lead_max_optype32', 'lead_max_optype256',
                       'lead_max_optype512', 'lead_max_optype64', 'optype12432', 'optype25651264']
        gamedf = gamedf.loc[:, gamedf_cols]
        # gamedf.drop(columns=['seat_order', 'leftcards_nums', 'startguid', 'uid', ], inplace=True)
        statistic_df = statistic_df.append(gamedf, sort=False, ignore_index=True)  # concat会匹配index,ignore_index

    # df = pd.concat([df, statistic_df], axis=1) # 按顺序匹配总是可能存在index的问题
    df = pd.merge(df, statistic_df, on='playtime_unix', copy=False)  # 根据出牌时间来匹配队友剩余牌数
    return df


def statistic_procedure_v2(df):
    """提取各情况的统计值第二版，对不同类型情况的分组统计"""

    # 出牌对手剩余手牌数
    def calculate_rival_leftcards_nums(x):
        if int(x) < 1:
            return "0"
        elif int(x) in [1, 3]:
            return "1,3"
        elif int(x) == 2:
            return "2"
        elif int(x) == 4:
            return "4"
        else:
            return ">4"

    df.loc[:, "rival_leftcards_nums"] = df.loc[:, 'rival_leftcards_nums'].astype(int)
    df.loc[:, "rival_leftcards_nums"] = df.loc[:, 'rival_leftcards_nums'].astype(str)
    df.loc[:, "rival_leftcards_nums"] = df.loc[:, 'rival_leftcards_nums'].apply(calculate_rival_leftcards_nums)

    # 出牌对手队友剩余手牌数
    def calculate_rival_leftcards_nums_pair(x):
        if int(x) < 1:
            return "0"
        elif int(x) < 5:
            return "1,2,3,4"
        elif int(x) < 11:
            return ">4,<11"
        elif int(x) < 16:
            return ">10,<16"
        else:
            return ">15"

    df.loc[:, "rival_leftcards_nums_pair"] = df.loc[:, 'rival_leftcards_nums_pair'].astype(int)
    df.loc[:, "rival_leftcards_nums_pair"] = df.loc[:, 'rival_leftcards_nums_pair'].astype(str)
    df.loc[:, "rival_leftcards_nums_pair"] = df.loc[:, 'rival_leftcards_nums_pair'].apply(
        calculate_rival_leftcards_nums_pair)

    # 队友剩余手牌数
    def calculate_leftcards_nums_pair(x):
        if int(x) < 1:
            return "0"
        elif int(x) < 4:
            return "1,2,3"
        elif int(x) < 16:
            return ">3,<16"
        else:
            return ">15"

    df.loc[:, "leftcards_nums_pair"] = df.loc[:, 'leftcards_nums_pair'].astype(int)
    df.loc[:, "leftcards_nums_pair"] = df.loc[:, 'leftcards_nums_pair'].astype(str)
    df.loc[:, "leftcards_nums_pair"] = df.loc[:, 'leftcards_nums_pair'].apply(calculate_leftcards_nums_pair)

    # 自己剩余牌信息
    def calculate_leftbomb_nums(x):
        if int(x) == 1:
            return "1"
        elif int(x) in [2, 3]:
            return "2,3"
        else:
            return ">3"

    df.loc[:, "leftbomb_nums"] = df.loc[:, 'leftbomb_nums'].astype(int)
    df.loc[:, "leftbomb_nums"] = df.loc[:, 'leftbomb_nums'].astype(str)
    df.loc[:, "leftbomb_nums"] = df.loc[:, 'leftbomb_nums'].apply(calculate_leftbomb_nums)

    # 手牌非炸弹手数
    df.loc[:, "Nonebomb_lefthands"] = df.loc[:, "Nonebomb_lefthands"].astype(int)
    df.loc[:, "Nonebomb_lefthands"] = df.loc[:, "Nonebomb_lefthands"].astype(str)
    df.loc[:, "Nonebomb_lefthands"] = df.loc[:, "Nonebomb_lefthands"].apply(lambda x: ">1" if int(x) > 1 else "1")

    # 手牌非炸平均牌值
    def calculate_onebomb_mean_cardvalue(x):
        if float(x) < 8.01:
            return "<=8"
        elif float(x) < 11.01:
            return ">8,<=11"
        else:
            return ">11"

    df.loc[:, "Nonebomb_MeanCardvalue"] = df.loc[:, 'Nonebomb_MeanCardvalue'].astype(str)
    df.loc[:, "Nonebomb_MeanCardvalue"] = df.loc[:, 'Nonebomb_MeanCardvalue'].apply(calculate_onebomb_mean_cardvalue)

    # 对手出牌情况
    def calculate_lead_cards(row):
        if int(row["lead_bomb"]) == 1:
            return 1  # 出炸
        elif int(row["lead_max_optype1"]) == 1:
            return 2  # "最大单张"
        elif int(row["lead_max_optype512"]) == 1:
            return 3  # "钢板"
        elif int(row["lead_max_optype256"]) == 1 or int(row["lead_max_optype64"]) == 1:
            return 4  # "连对or顺子"
        elif int(row["lead_max_optype2"]) == 1 or int(row["lead_max_optype4"]) == 1 or int(
                row["lead_max_optype32"]) == 1:
            return 5  # 最大对子，三张，三带二
        else:
            return 0  # 其他情况

    df.loc[:, "lead_cards"] = df.apply(lambda row: calculate_lead_cards(row), axis=1)

    # 处理房间号：从startguid中拆分出来
    def room_sep(x):
        if x in ["17743", '17744', '8136', '10321', '10163', '18934', '9533']:
            return "经典积分"
        else:
            return "不洗牌"

    df.loc[:, 'startguid'] = df.loc[:, 'startguid'].apply(lambda x: x.split("_")[1])
    df.loc[:, 'startguid'] = df.loc[:, 'startguid'].apply(room_sep)

    # 筛选数据进行统计
    used_cols = ["startguid", 'rival_leftcards_nums', 'rival_leftcards_nums_pair', 'leftcards_nums_pair',
                 'leftbomb_nums', 'Nonebomb_lefthands', 'lead_cards', 'need_bomb', 'label_bomb']
    statistic_df = df.loc[
        (df.loc[:, 'need_bomb'] == 1) & (df.loc[:, 'lead_cards'] > 0) & (df.loc[:, 'label_uid'] > 0),
        used_cols]
    statistic_df.reset_index(drop=True, inplace=True)
    data_length = statistic_df.startguid.nunique()
    if statistic_df.shape[0]:
        statistic_df.loc[:, 'need_bomb'] = statistic_df.loc[:, 'need_bomb'].astype(int)
        statistic_df.loc[:, 'label_bomb'] = statistic_df.loc[:, 'label_bomb'].astype(int)
        statistic_df = statistic_df.groupby(used_cols[:-2]).agg({
            'need_bomb': np.sum,
            'label_bomb': np.sum,
        })
        statistic_df.reset_index(drop=False, inplace=True)
        statistic_df.loc[:, "lead_cards"] = statistic_df.loc[:, 'lead_cards'].astype(str)
        lead_cards_name_dict = {"1": "出炸", "2": "最大单张", "3": "钢板", "4": "连对或顺子", "5": "最大对子、三张、三带二"}
        statistic_df.loc[:, "lead_cards"] = statistic_df.loc[:, 'lead_cards'].map(lead_cards_name_dict)

        robot_bomb_statistic_situations = {
            "Nonebomb_lefthands": "手牌非炸手数",
            "Nonebomb_MeanCardvalue": "手牌非炸平均牌值",
            "lead_cards": "对手出牌",
            "leftbomb_nums": "自己剩余炸弹数",
            "leftcards_nums_pair": "队友剩余手牌数",
            "rival_leftcards_nums_pair": "出牌对手队友剩余手牌数",
            "rival_leftcards_nums": "出牌对手剩余手牌数",
            "need_bomb": "occurrence_times",
            "label_bomb": "lead_bomb_times",
            'startguid': "房间号",
        }
        statistic_df.rename(columns=robot_bomb_statistic_situations, inplace=True)

        statistic_df.loc[:, 'probability'] = statistic_df.apply(
            lambda row: round(row["lead_bomb_times"] / row["occurrence_times"], 4)
            if row["occurrence_times"] != 0 else 0, axis=1)

        # 存储统计结果
        write_data(statistic_df, filedir=os.path.join(tmpdatadir1, "20190625"), filename='statistic_result', )
    return statistic_df, data_length


def do_statistic_procedure(filepath, prefix='robot_result', grouper=False, out_name='statistic_result'):
    """
    对拆牌后打完标记的多个明细数据文件进行统计数据提取
    :param filepath: 明细标记数据 目录
    :param prefix: 明细标记数据文件前缀
    :param grouper: 是否合并多个 统计数据结果（合并处理还是分开处理）
    :param out_name: 输出文件名，（前缀）
    :return:grouper=True，则输出一个总的统计文件，grouper=False则输出多个统计文件，和一个总的统计文件
    """
    data = pd.DataFrame()
    allfiles = [file for file in os.listdir(filepath) if file.startswith(prefix)]
    for file in allfiles:
        df = pd.read_csv(os.path.join(filepath, file), encoding='gbk')
        df, _ = statistic_procedure_v2(df)
        if df.shape[0]:
            if grouper:
                data = pd.concat([data, df], sort=False)
            else:
                out_file_name = f"{out_name}-{file.split('.')[0]}"
                write_data(df, filedir=filepath, filename=out_file_name)

    if grouper:
        data = data.groupby(list(data.columns)[:-3]).agg({
            'occurrence_times': np.sum,
            'lead_bomb_times': np.sum,
        })
        data.loc[:, 'probability'] = data.apply(
            lambda row: round(
                row["lead_bomb_times"] / row["occurrence_times"], 4) if row["occurrence_times"] != 0 else 0, axis=1)
        data = data.reset_index(drop=False)
        write_data(data, filedir=filepath, filename=f"all_{out_name}")
    else:
        statistic_files = [file for file in os.listdir(filepath) if file.startswith(out_name)]
        for statistic_file in statistic_files:
            statistic_df = pd.read_csv(os.path.join(filepath, statistic_file), encoding='gbk')
            data = pd.concat([data, statistic_df], sort=False)

        data = data.groupby(list(data.columns)[:-3]).agg({
            'occurrence_times': np.sum,
            'lead_bomb_times': np.sum,
        })
        data.loc[:, 'probability'] = data.apply(
            lambda row: round(
                row["lead_bomb_times"] / row["occurrence_times"], 4) if row["occurrence_times"] != 0 else 0, axis=1)
        data = data.reset_index(drop=False)
        write_data(data, filedir=filepath, filename=f"all_{out_name}", index=False)


def circle_statistic_procedure(df, filepath, data_length, all_data_length):
    """
    df 当前统计数据结果
    filepath 确定文件路径
    data_length 局数
    all_data_length 总对局数"""
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    files = [file for file in os.listdir(filepath) if file.startswith("latest")]
    if files:
        previous_data_length = int(files[-1].split("_")[1])
        previous_data_length = previous_data_length + data_length
        previous_data = pd.read_csv(os.path.join(filepath, files[-1]), encoding='gbk')
        # col_names = list(previous_data.columns)
        df = pd.concat([previous_data, df], axis=0, sort=False, ignore_index=True)
        # 合并之前的总数据和现得数据
        df = df.groupby(list(df.columns)[:-3]).agg({
            'occurrence_times': np.sum,
            'lead_bomb_times': np.sum,
        })
        # df.drop(columns=['probability'], inplace=True)
        df.loc[:, 'probability'] = df.apply(
            lambda row: round(
                row["lead_bomb_times"] / row["occurrence_times"], 4) if row["occurrence_times"] != 0 else 0, axis=1)
        df = df.reset_index(drop=False)
    else:
        previous_data_length = data_length

    for file in files:
        os.remove(os.path.join(filepath, file))
    write_data(df, filedir=filepath, filename=f"latest_{previous_data_length}_{all_data_length}", index=False)


def main_process(process_test=True, win_ratio=0.5, data_sep=10000):
    first_init()   # 初始化配置
    if process_test:
        get_raw_test_data(True,
                          ["81_17743_3_1558830025_4",
                           '81_18149_48_1558830417_4',
                           '81_8136_60_1558830808_4',
                           '81_10163_4_1558830693_4',
                           '81_18149_13_1558811750_4',
                           '81_18149_13_1558811183_4',
                           '81_17743_42_1558834539_4',
                           '81_17743_42_1558833612_4'])  # 生成测试数据
        mergedata = get_raw_test_data()  # 读取测试数据

    else:
        mergedata = get_raw_data(win_ratio=win_ratio)  # 读取正式数据

    def combine_result_files(result_filepath):
        if os.path.exists(result_filepath):
            #  标记结果合并
            robot_result = pd.DataFrame()
            current_result_dir = os.path.join(tmpdatadir1, "20190625")
            robot_result_files = [file for file in os.listdir(current_result_dir) if file.startswith("robot_result")]
            if robot_result_files:
                for file in robot_result_files:
                    filepath = os.path.join(current_result_dir, file)
                    data_file = pd.read_csv(filepath, encoding='gbk', header=0)
                    robot_result = pd.concat([robot_result, data_file], axis=0, sort=False, copy=False)
                    os.remove(filepath)
                write_data(robot_result, filedir=result_filepath, filename='robot_result')
            #  统计结果合并
            statistic_result = pd.DataFrame()
            statistic_result_files = [file for file in os.listdir(current_result_dir) if
                                      file.startswith("statistic_result")]
            if statistic_result_files:
                for file in statistic_result_files:
                    filepath = os.path.join(current_result_dir, file)
                    data_file = pd.read_csv(filepath, encoding='gbk', header=0)
                    statistic_result = pd.concat([statistic_result, data_file], axis=0, sort=False, copy=False)
                    os.remove(filepath)
                write_data(statistic_result, filedir=result_filepath, filename='statistic_result')
        else:
            os.mkdir(result_filepath)
            combine_result_files(result_filepath)

    unique_startguid = mergedata.loc[:, "startguid"].unique()
    unique_startguid_length = len(unique_startguid)
    if unique_startguid_length < data_sep:
        mergedata = basic_treatment(mergedata)  # 初步处理
        mergedata = calculate_cards_value(mergedata)  # 拆牌结果
        statistic_procedure_v2(mergedata)  # 统计结果
    else:
        # latest_result_files = [file for file in os.listdir(outdatadir)]
        # if latest_result_files:
        #     for file in latest_result_files:
        #         shutil.copy2(os.path.join(outdatadir, file), outdatadir1)  # 对最新数据结果做阶段性备份
        sep_bins = list(range(0, unique_startguid_length+data_sep, data_sep))
        sep_bins_length = len(sep_bins)
        for start_index in range(sep_bins_length-1):
            chunk_df = mergedata.loc[
                mergedata.loc[:, 'startguid'].isin(unique_startguid[sep_bins[start_index]:sep_bins[start_index + 1]])]
            chunk_df.reset_index(drop=True, inplace=True)
            chunk_df = basic_treatment(chunk_df)  # 初步处理
            chunk_df = calculate_cards_value(chunk_df)  # 拆牌结果
            # write_data(chunk_df, filename='aaaaaaaaa')
            chunk_df, data_length = statistic_procedure_v2(chunk_df)  # 前置，统计结果
            if chunk_df.shape[0]:
                # 生成latest版本统计结果
                circle_statistic_procedure(chunk_df, os.path.join(tmpdatadir1, '20190625'),
                                           data_length=data_length, all_data_length=sep_bins[start_index + 1])
            if start_index % 1000 == 0:
                combine_result_files(os.path.join(tmpdatadir1, '20190625', 'detail_result'))
            if start_index == sep_bins_length-1:
                combine_result_files(os.path.join(tmpdatadir1, '20190625', 'detail_result'))
            # del chunk_df

    # 后置，统计数据提取
    # do_statistic_procedure(tmpdatadir1, prefix='robot_result', grouper=False, out_name="statistic_result")


if __name__ == '__main__':
    reduce_raw_data()  # 缩减原始数据体积
    # main_process(True, data_sep=4)  # 测试数据
    main_process(process_test=False, win_ratio=0.5, data_sep=1, )
