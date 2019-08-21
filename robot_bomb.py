# -*- encoding: utf-8 -*-
# @Version : 4
# @Time    : 2019/6/5 14:05 V1
# @Time    : 2019/7/3 13:46 V2
# @Time    : 2019/7/24 16:05 V3 拆牌、压牌逻辑变更
# @Time    : 2019/8/16 11:15 V4 修复对局日志选手跑完牌后无记录，导致选手错位，以致统计标记错误的问题
# @Author  : wanghd
# @note    : 用户牌局出炸情况处理（拆牌+ 统计）


import os
# import shutil
import sys
import time
import configparser
from collections import Counter

import numpy as np
import pandas as pd
import subprocess
from itertools import compress
from tqdm import tqdm


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


def write_data(df, filedir=os.path.join(os.getcwd(), 'tmpdata1'), filename='robot_bomb', index=False):
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
        show.rename(columns={"num": "num_show"}, inplace=True)
        cus['cards_order'] = 1
        mergedata = pd.merge(show, cus, on=['startguid', 'uid', 'cards_order'], how='left', copy=False)
        return mergedata


def get_raw_data(win_ratio=0.5):
    """读取正式数据"""
    # cus_usecols = ['startguid', 'uid', 'cards', 'num','rank'] 起手牌牌id[0-107], 起手牌牌数字[1-14]
    cus_usecols = ['startguid', 'uid', 'cards', 'rank', 'label_uid', 'num']  # 起手牌，级牌，胜率大于要求的用户uid标记
    cus_files = [file for file in os.listdir(rawdatadir) if
                 file.startswith(f'short_win_ratio_{win_ratio}_customization_20190709')]
    print(cus_files)
    name_dict = {'cards': "cards_init", 'num': 'num_init'}  # 重命名为 初始牌组

    # 对局id,uid, 开局时间，每手出牌时间点，出牌状态（出，不出，托管）'optype',
    # 牌型type，出牌回合, 出牌信息id[0-107]，出牌数字信息[1,14]
    # ['startguid', 'uid', 'starttime_unix', 'playtime_unix', 'type', 'cards_order', 'cards', 'num']
    show_usecols = ['startguid', 'uid', 'starttime_unix', 'playtime_unix', 'type', 'cards_order', 'cards', 'num']
    show_files = [file for file in os.listdir(rawdatadir) if
                  file.startswith(f'short_win_ratio_{win_ratio}_showcards_20190709')]
    print(show_files)
    for cus_file, show_file in zip(cus_files, show_files):
        cus = pd.read_csv(os.path.join(rawdatadir, cus_file), usecols=cus_usecols, encoding='gbk')  # 读取定制信息
        cus.rename(columns=name_dict, inplace=True)  # 重命名为初始牌组
        cus['cards_order'] = 1  # 起手牌标记出牌回合为 1
        show = pd.read_csv(os.path.join(rawdatadir, show_file), usecols=show_usecols, encoding='gbk')  # 读取出牌信息
        show.rename(columns={"num": "num_show"}, inplace=True)

        # 将初始牌组 和 出牌信息合并
        show = pd.merge(show, cus, on=['startguid', 'uid', 'cards_order'], how='left', copy=False)
        del cus
        return show


def get_config_dict(filepath, filename, section_name):
    """读取配置文件中的字典"""
    config = configparser.ConfigParser()
    config.read(os.path.join(filepath, filename))
    config_dict = dict(config.items(section_name))
    # print(config_dict)
    return config_dict


def reduce_raw_data(win_ratio=0.5):
    """缩减原始数据体积，仅保存需要的列， 筛选存在用户胜率大于等于50%的对局"""
    first_init()
    # 缩减customization 定制表体积
    cus_usecols = ['startguid', 'uid', 'cards', 'rank', 'num']  # 起手牌,级牌，起手牌牌面信息
    cus_files = [file for file in os.listdir(rawdatadir) if file.startswith('customization_20190709')]
    # 缩减showcards 出牌信息表体积
    show_usecols = ['startguid', 'uid', 'starttime_unix', 'playtime_unix', 'type', 'cards_order', 'cards', 'num']
    show_files = [file for file in os.listdir(rawdatadir) if file.startswith('showcards_20190709')]
    # 缩减start开局表，有历史对局战绩信息
    start_usecols = ['uid', 'win', 'loss']  # 历史胜局数，历史输局数
    start_files = [file for file in os.listdir(rawdatadir) if file.startswith('start_20190709')]

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


def card_id2card_num(card_id):
    """将牌ID值 转化为 牌面值"""
    temp_list = []
    for card in card_id:
        try:
            card54id = int(card) % 54
            if card54id < 52:
                if card54id % 13 < 12:
                    temp_list.append(card54id % 13 + 2)
                else:
                    temp_list.append(1)
            elif card54id == 52:
                temp_list.append(14)  # 小王
            else:
                temp_list.append(15)  # 牌ID 对应 牌面中 大王也是14，这里特殊处理
        except ValueError:
            temp_list.append(-1)  # 应对可能的空值
    return sorted(temp_list)


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
    mergedata.loc[:, 'num_init'] = mergedata.loc[:, 'num_init'].fillna(method='ffill')  # 初始牌牌面信息
    # 整理累积出牌信息
    mergedata.loc[:, 'cum_cards'] = mergedata.loc[:, 'cum_cards'].astype(str)
    mergedata.loc[:, 'cum_cards'] = mergedata.loc[:, 'cum_cards'].apply(lambda x: set(str(x[:-1]).split(sep=',')))
    # 整理初始牌信息
    mergedata.loc[:, 'cards_init'] = mergedata.loc[:, 'cards_init'].apply(lambda x: set(str(x).split(sep=',')))

    # 计算剩余牌信息 ID组
    mergedata.loc[:, 'leftcards'] = mergedata.apply(lambda row: row['cards_init'] - row['cum_cards'], axis=1)
    mergedata.loc[:, 'leftcards_face'] = mergedata.loc[:, 'leftcards'].apply(card_id2card_num)  # 剩余牌牌面信息
    # 剩余牌数量
    mergedata.loc[:, 'leftcards_nums'] = mergedata.loc[:, 'leftcards'].apply(lambda x: len(x))
    mergedata.loc[:, 'leftcards_str'] = mergedata.loc[:, 'leftcards'].apply(lambda x: "".join(sorted(list(x))))
    mergedata.drop(columns=['cum_cards'], inplace=True)  # 删除累积出牌信息

    # write_data(mergedata, filename='robot_basic')

    return mergedata


def calculate_lead_cards_value(df):
    """调用拆牌程序取 出牌 的牌值"""
    base_path = os.path.abspath("F:/CardsValue")
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
    subprocess.call(os.path.join(base_path, 'CalculatorCardsValue1dir.exe'))

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
    subprocess.call(os.path.join(base_path, 'CalculatorCardsValue1dir.exe'))

    # 读取拆牌信息
    df_cardsvalue_cols = ['leftcards_str', 'cards_value', 'cards_id', 'cards_type']  # 唯一标识，牌力值，ID组，类型组
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
                    df_cardsvalue.at[index - 1, 'leftcards_str'] = line.strip()
                if index % 5 == 3:
                    df_cardsvalue.at[index - 3, 'cards_value'] = line.strip()
                if index % 5 == 4:
                    df_cardsvalue.at[index - 4, 'cards_id'] = line.strip()
                if index % 5 == 0:
                    df_cardsvalue.at[index - 5, 'cards_type'] = line.strip()
                # print(index, line)

    df_cardsvalue.drop_duplicates(inplace=True)
    # 存储拆牌前的数据
    # write_data(df, filename='robot_001_notchaipai')

    df = pd.merge(df, df_cardsvalue, on=['leftcards_str'], how='left', copy=False)  # 合并牌力值拆牌信息
    df.drop(columns=["leftcards_str"], inplace=True)  # 删除合并标识列 leftcards_str

    df = calculate_lead_cards_value(df)  # 出牌牌值计算
    df = apart_cards_type(df)  # 牌型拆解, 牌型信息标记
    df = rival_leadcards_treatment(df)  # 提取对手出牌，各出牌情况标记，对手剩余牌

    # 存储拆牌后的数据
    # df.rename(columns=get_config_dict(os.getcwd(), 'robot_bomb_colnames.cfg', 'colname'), inplace=True)
    write_data(df, filedir=os.path.join(tmpdatadir1, '20190709'), filename='robot_result')

    return df


def apart_cards_type(df):
    """
    牌型拆解,
    :param df: 经过牌力值程序计算之后带有 牌型 的 dataframe
    :return: 各牌型数量，剩余手牌中各牌型的最大牌值，
    """

    # 单，对，三，顺，连对，钢板，炸45，同花，超炸6-10,4王
    cardoptypelist = [1, 2, 4, 64, 256, 512, 4096, 16384, 32768, 524288]

    def apart_card(x):
        """拆分牌组，取各牌型的数量 + 最大牌值"""
        # 1,2,4,64,256,512,4096,16384,32768,524288
        # 单，对，三，顺，连对，钢板，炸45，同花，超炸6-10,4王
        # cardoptypelist = [1, 2, 4, 64, 256, 512, 4096, 16384, 32768, 524288]
        cards_list = str(x).split(sep='||')
        if 'nan' not in cards_list:
            card_optype_list = [card.split(sep='|')[0] for card in cards_list]  # 牌类型
            # card_nums_list = [card.split(sep='|')[1] for card in cards_list]  # 牌数量
            card_value_list = [card.split(sep='|')[2] for card in cards_list]  # 牌值
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

    def get_leftcards_exclude(row):
        """获得除炸弹顺子外剩余手牌ID组"""
        paixing_list = str(row["cards_type"]).split(sep='||')
        leftcards_exclude = '0'
        if 'nan' not in paixing_list:
            card_optype_list = [card.split(sep='|')[0] for card in paixing_list]  # 牌类型
            boolen = [card_type in ["1", '2', '4', '256', '512'] for card_type in card_optype_list]
            if sum(boolen) > 0:
                card_id_list = str(row["cards_id"]).split(sep='||')
                leftcards_exclude_list = compress(card_id_list, boolen)  # 筛选出非炸非顺子手牌
                leftcards_exclude = ",".join(
                    [cards_id for cards in leftcards_exclude_list for cards_id in cards.split(sep="|")])
        return leftcards_exclude

    # 产生剩余手牌中除炸除顺子外牌ID组
    df.loc[:, "leftcards_exclude"] = df.apply(get_leftcards_exclude, axis=1)

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
                card_type, card_num, card_value = card.split(sep='|')
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


def rival_leadcards_treatment(df):
    """提取需要的统计数据的基础数据"""
    # 循环处理每一局
    df.loc[:, 'cards'] = df.loc[:, 'cards'].fillna("0")  # 手牌ID组
    df.loc[:, 'num_show'] = df.loc[:, 'num_show'].fillna('0')  # 手牌牌面
    df.loc[:, "cards_value"] = df.loc[:, "cards_value"].fillna(0)  # 剩余牌牌力值
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
            compare_df.at[compare_idx + plus_idx, 'need_bomb'] = 1  # 需要出炸
            if compare_df.at[compare_idx + plus_idx, 'type'] > 0:
                compare_df.at[compare_idx + plus_idx, 'label_bomb'] = 1  # 出炸
        # 类型相同比牌数
        elif inhand_bomb_max_type == compare_df.at[compare_idx, 'type']:
            bomb_card_nums = len(str(compare_df.at[compare_idx, 'cards']).split(","))  # 出牌炸弹的牌数
            if compare_df.at[compare_idx + plus_idx, 'leftmaxbomb_cardnum'] > bomb_card_nums:
                compare_df.at[compare_idx + plus_idx, 'need_bomb'] = 1  # 需要出炸
                if compare_df.at[compare_idx + plus_idx, 'type'] > 0:
                    compare_df.at[compare_idx + plus_idx, 'label_bomb'] = 1  # 出炸
            # 牌数相同比牌值
            elif compare_df.at[compare_idx + plus_idx, 'leftmaxbomb_cardnum'] == bomb_card_nums:
                if inhand_bomb_max_cardvalue > int(compare_df.at[compare_idx, 'leadcards_cardvalue']):
                    compare_df.at[compare_idx + plus_idx, 'need_bomb'] = 1  # 需要出炸
                    if compare_df.at[compare_idx + plus_idx, 'type'] > 0:
                        compare_df.at[compare_idx + plus_idx, 'label_bomb'] = 1  # 出炸
        return compare_df

    def get_rival_info(info_df, info_idx, plus_idx, game_start=False):
        """取出牌对手信息：出牌信息，剩余牌信息，位置信息
        :param info_df: 源数据df
        :param info_idx: 出牌对手的 idx
        :param plus_idx: 记录出牌对手信息的用户的index 对于出牌对手index 的偏移量（可代表上家、下家）
        :param game_start: 是否是开局的时候（开局时候取出牌对手队友剩余牌信息比较特殊）
        """
        info_df.at[info_idx + plus_idx, 'rival_leftcards_nums'] = info_df.at[info_idx, 'leftcards_nums']  # 出牌对手
        if game_start:
            info_df.at[info_idx + plus_idx, 'rival_leftcards_nums_pair'] = 27
        else:
            info_df.at[info_idx + plus_idx, 'rival_leftcards_nums_pair'] = info_df.at[
                info_idx - 2, 'leftcards_nums']  # 出牌对手队友剩余牌数量
        info_df.at[info_idx + plus_idx, 'rival_leadcards_type'] = info_df.at[info_idx, 'type']  # 对手出牌类型
        info_df.at[info_idx + plus_idx, 'rival_leadcards_cards'] = info_df.at[info_idx, 'cards']  # 对手出牌ID组
        info_df.at[info_idx + plus_idx, 'rival_leadcards_num_show'] = info_df.at[info_idx, 'num_show']  # 对手出牌牌面
        info_df.at[info_idx + plus_idx, 'rival_cards_value'] = info_df.at[info_idx, 'cards_value']  # 对手剩余牌力值
        info_df.at[info_idx + plus_idx, 'rival_position'] = plus_idx % 3  # 出牌对手位置 1 上家，0 下家
        return info_df

    def compare_leftcards_exclude(source_df, source_idx, plus_idx):
        # 对手出牌信息
        leadcards_cardvalue = int(source_df.at[source_idx, "leadcards_cardvalue"])  # 对手出牌牌值
        leadcards_type = source_df.at[source_idx, 'type']  # 对手出牌类型

        # 除炸除顺子外剩余手牌
        leftcards_exclude = set(str(source_df.at[source_idx+plus_idx, 'leftcards_exclude']).split(","))
        if len(leftcards_exclude) > 0:
            leftcards_exclude_face = Counter(card_id2card_num(leftcards_exclude))  # 牌面
        else:
            leftcards_exclude_face = Counter([0])

        # rank列记录到级牌ID的偏移量
        jipai_id_offset = [-1, 12, 25, 38, 53, 66, 79, 92]
        try:
            jipai_rank = int(source_df.at[source_idx, 'rank'])
        except ValueError:
            jipai_rank = 1  # 如果缺失，默认级牌为 2 （ID=1）
        jipai_id = {str(jipai_rank + offset) for offset in jipai_id_offset}  # 级牌ID
        jipai_heart_id = {str(jipai_rank + 25), str(jipai_rank + 79)}  # 红桃级牌ID

        # 剩余牌级牌数量， 红桃级牌数量
        inter = leftcards_exclude.intersection(jipai_id)
        jipai_nums = len(inter)  # 级牌数量
        inter_heart = inter.intersection(jipai_heart_id)
        jipai_heart_nums = len(inter_heart)  # 红桃级牌数量
        leftcards_exclude_jipai = leftcards_exclude - inter  # 除级牌外剩余手牌

        leftcards_exclude_jipai_face = Counter(card_id2card_num(leftcards_exclude_jipai))  # 除级牌外 剩余牌牌面
        paimian = list(leftcards_exclude_jipai_face.keys())  # 除级牌外
        paimian_nums = list(leftcards_exclude_jipai_face.values())  # 除级牌外

        if leadcards_type in [1]:
            # 出单张的情况
            if leadcards_cardvalue in [40, 41, 42]:
                # 出级牌 或者 小王
                if sum([x > (leadcards_cardvalue - 27) for x in paimian]) > 0:
                    # 是否存在牌面大于出牌牌面的数值
                    source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
            else:
                # 对于非级牌的牌，牌面就是牌值+1 ； 此时判断是否存在比牌面值大的牌或者有级牌
                if (sum([x > (leadcards_cardvalue + 1) for x in paimian]) > 0) or (jipai_nums > 0):
                    # 是否存在牌面大于出牌牌面的数值
                    source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
        elif leadcards_type in [2]:
            # 出对子的情况
            if leadcards_cardvalue in [40, 41, 42]:
                # 出级牌 或者 小王
                bool_list = [x > (leadcards_cardvalue - 27) for x in paimian]
                if sum(bool_list) > 0 and sum([x > 1 for x in compress(paimian_nums, bool_list)]) > 0:
                    # 是否存在牌面大于出牌牌面的数值
                    source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
            elif leadcards_cardvalue in [13]:
                # 出A
                bool_list = [x > leadcards_cardvalue for x in paimian]
                if (sum(bool_list) > 0 and sum(
                        [x > 1 for x in compress(paimian_nums, bool_list)]) > 0) or (jipai_nums > 1):
                    # 出小王,大王, 或者级牌
                    source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
            else:
                # 对于出牌为 非级牌，非王，非A 的牌
                if jipai_heart_nums < 1:
                    # 不存在红桃级牌
                    bool_list = [x > (leadcards_cardvalue + 1) for x in paimian]
                    if (sum(bool_list) > 0 and sum(
                            [x > 1 for x in compress(paimian_nums, bool_list)]) > 0) or (jipai_nums > 1):
                        # 是否存在牌面大于出牌牌面的数值
                        source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
                else:
                    # 存在红桃级牌
                    bool_list = [x > (leadcards_cardvalue + 1) for x in paimian]
                    if (sum(bool_list) > 0) or (jipai_nums > 1):
                        # 是否存在牌面大于出牌牌面的数值
                        source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
        elif leadcards_type in [4]:
            # 出三张的情况
            if leadcards_cardvalue in [40, 41, 42]:
                # 出级牌 或者 小王，大王
                pass
            elif leadcards_cardvalue in [13]:
                # 出A
                if jipai_nums > 2:
                    # 压级牌
                    source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
            else:
                # 对于出牌为 非级牌，非王，非A 的牌
                if jipai_heart_nums < 1:
                    # 不存在红桃级牌
                    bool_list = [x > (leadcards_cardvalue + 1) for x in paimian]
                    if (sum(bool_list) > 0 and sum(
                            [x > 2 for x in compress(paimian_nums, bool_list)]) > 0) or (jipai_nums > 2):
                        # 大三张 或者 3级牌
                        source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
                elif jipai_heart_nums < 2:
                    # 存在 1 张红桃级牌
                    bool_list = [x > (leadcards_cardvalue + 1) for x in paimian]
                    if (sum(bool_list) > 0 and sum(
                            [x > 1 for x in compress(paimian_nums, bool_list)]) > 0) or (jipai_nums > 2):
                        # 大对子配1红桃级牌 或 3级牌
                        source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
                else:
                    # 存在 2 张红桃级牌
                    bool_list = [x > (leadcards_cardvalue + 1) for x in paimian]
                    if (sum(bool_list) > 0 and sum(
                            [x > 0 for x in compress(paimian_nums, bool_list)]) > 0) or (jipai_nums > 2):
                        # 大单配2红桃级牌 或 3级牌
                        source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
                    elif sum(bool_list) > 0 and sum(
                            [x > 1 for x in compress(paimian_nums, bool_list)]) > 0:
                        # 大对子配1红桃级牌
                        source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
        elif leadcard_type in [32]:
            if int(source_df.at[source_idx, 'leadcards_cardvalue']) >= int(
                    source_df.at[source_idx + plus_idx, 'optype4_maxcardvalue']):
                # 三张压不住的情况
                if leadcards_cardvalue in [40, 41, 42]:
                    # 出级牌 或者 小王，大王
                    source_df.at[source_idx + plus_idx, 'need_bomb'] = 1
                elif leadcards_cardvalue in [13]:
                    if (jipai_nums > 2) and (len([x > 1 for x in paimian_nums]) > 0):
                        # 出级牌 + 其他对子
                        source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
                    else:
                        source_df.at[source_idx + plus_idx, 'need_bomb'] = 1  # 不可压
                else:
                    # 对于出牌为 非级牌，非A 的牌
                    if jipai_heart_nums < 1:
                        # 不存在红桃级牌
                        bool_list = [x > (leadcards_cardvalue + 1) for x in paimian]
                        if (sum(bool_list) > 0 and sum(
                                [x > 2 for x in compress(paimian_nums, bool_list)]) > 0) and (
                                len([(k, v) for k, v in leftcards_exclude_face.items() if v > 1]) > 1):
                            # 是否存在牌面大于出牌牌面的数值, 并且有对子
                            source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
                        elif (jipai_nums > 2) and (
                                len([(k, v) for k, v in leftcards_exclude_face.items() if v > 1]) > 1):
                            source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
                        else:
                            source_df.at[source_idx + plus_idx, 'need_bomb'] = 1  # 不可压
                    elif jipai_heart_nums < 2:
                        # 存在 1 张红桃级牌
                        bool_list = [x > (leadcards_cardvalue + 1) for x in paimian]
                        if (sum(bool_list) > 0 and sum(
                                [x > 1 for x in compress(paimian_nums, bool_list)]) > 0) and (
                                (len([(k, v) for k, v in leftcards_exclude_jipai_face.items() if v > 1]) > 1) or (
                                jipai_nums > 2)):
                            # 非级牌牌面+红桃级牌=三张  配 其他对子或者级牌对子
                            source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
                        elif (jipai_nums > 2) and (
                                len([(k, v) for k, v in leftcards_exclude_jipai_face.items() if v > 1]) > 1):
                            # 三张级牌 + 其他对子
                            source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
                        elif (jipai_nums > 3) and (
                                len([(k, v) for k, v in leftcards_exclude_face.items() if v > 1]) > 0):
                            # 三张级牌 + 单牌配红桃级牌
                            source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
                        else:
                            source_df.at[source_idx + plus_idx, 'need_bomb'] = 1  # 不可压
                    else:
                        # 存在 2 张红桃级牌
                        bool_list = [x > (leadcards_cardvalue + 1) for x in paimian]
                        if (sum(bool_list) > 0 and sum(
                                [x > 0 for x in compress(paimian_nums, bool_list)]) > 0) and (
                                (len([(k, v) for k, v in leftcards_exclude_jipai_face.items() if v > 1]) > 1) or (
                                jipai_nums > 3)):
                            # 大单配2红桃级牌 + 其他对子或2级牌
                            source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
                        elif (sum(bool_list) > 0 and sum(
                                [x > 1 for x in compress(paimian_nums, bool_list)]) > 0) and (
                                (len([(k, v) for k, v in leftcards_exclude_jipai_face.items() if v > 0]) > 1) or (
                                jipai_nums > 2)):
                            # 大对子配1红桃级牌 + 其他单牌配1红桃级牌 或 2级牌
                            source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
                        elif (jipai_nums > 2) and (
                                len([(k, v) for k, v in leftcards_exclude_jipai_face.items() if v > 1]) > 0):
                            # 3级牌 + 1对子
                            source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
                        elif (jipai_nums > 3) and (
                                len([(k, v) for k, v in leftcards_exclude_jipai_face.items() if v > 0]) > 0):
                            # 3 级牌 + 单配1红桃级牌
                            source_df.at[source_idx + plus_idx, 'need_bomb'] = 0  # 可压
                        else:
                            source_df.at[source_idx + plus_idx, 'need_bomb'] = 1  # 不可压
            else:
                # 三张压的住，但是没有对子
                if (int(source_df.at[source_idx + plus_idx, 'optype2_nums']) < 1) and (
                        source_df.at[source_idx + plus_idx, 'leftbomb_nums'] > 0):
                    if len([(k, v) for k, v in leftcards_exclude_face.items() if v > 1]) > 1:
                        # 剩余手牌中大于等于2张的牌面种类数 ，多于1 可压，否则不可压(三张压的住的时候红桃级牌不变牌)
                        source_df.at[source_idx + plus_idx, 'need_bomb'] = 0
                    else:
                        source_df.at[source_idx + plus_idx, 'need_bomb'] = 1
            # 所有情况都需要判断最终是否出了炸
            if source_df.at[source_idx + plus_idx, 'type'] >= 4096:
                source_df.at[source_idx + plus_idx, 'label_bomb'] = 1  # 出炸
        return source_df

    def compare_cards(type_df, type_card, source_idx, plus_idx, stage=False):
        """
        :param type_df: 源df
        :param type_card: 出牌牌型
        :param source_idx: 出牌方 index
        :param plus_idx: 己方相对出牌方的index offset
        :param stage: 是否需要继续判断除炸除顺子外剩余牌是否可压
        :return:
        """
        type_card_col = f"optype{type_card}_maxcardvalue"  # 当前对手出牌牌型，在我手中的最大牌值
        if int(type_df.at[source_idx, 'leadcards_cardvalue']) >= int(type_df.at[source_idx + plus_idx, type_card_col]):
            # 牌力值牌型压不住的情况
            if type_df.at[source_idx + plus_idx, 'leftbomb_nums'] > 0:
                type_df.at[source_idx + plus_idx, 'need_bomb'] = 1
                if stage:
                    # 判断剩余牌是否可压
                    type_df = compare_leftcards_exclude(type_df, source_idx, plus_idx)
                    if type_df.at[source_idx + plus_idx, 'type'] >= 4096:
                        type_df.at[source_idx + plus_idx, 'label_bomb'] = 1  # 出炸
                else:
                    if type_df.at[source_idx + plus_idx, 'type'] >= 4096:
                        type_df.at[source_idx + plus_idx, 'label_bomb'] = 1  # 出炸
        return type_df

    for start_guid in start_guids:
        gamedf = df.loc[df.loc[:, 'startguid'] == start_guid]
        cards_order_df = pd.DataFrame(list(range(1, gamedf.cards_order.max()+1)), columns=["cards_order"])
        cards_order_df.loc[:, 'startguid'] = start_guid
        cards_order_seat_order = gamedf.loc[:, ['startguid', 'uid', 'seat_order', "playtime_unix"]].sort_values(by=["playtime_unix"], ascending=True)
        cards_order_seat_order = cards_order_seat_order.groupby(["startguid"]).head(4)
        cards_order_seat_order.drop(columns=["playtime_unix"], inplace=True)
        cards_order_df = pd.merge(cards_order_df, cards_order_seat_order, on=['startguid'], how='left')

        # 输出自构建的出牌回合df
        # cards_order_df.to_excel(f"F:/aaa/cards_order_df.xlsx", index=False)

        # print(cards_order_df.head())
        gamedf.drop(columns=["seat_order"], inplace=True)  # 删除原始的seat_order
        gamedf = pd.merge(gamedf, cards_order_df, on=["startguid", 'cards_order', 'uid'], how='right')

        # 输出合并自构建出牌回合和座位号df之后的 gamedf
        # gamedf.to_excel(f"F:/aaa/gamedf.xlsx", index=False)

        gamedf = gamedf.sort_values(by=['uid', 'cards_order'], ascending=[True, True])
        gamedf.reset_index(drop=True, inplace=True)
        # 填充由于填补 cards_order 导致的缺失值
        for col in ["cards", 'num_show', 'leftcards_exclude','cards_id','cards_type','leftcards_exclude']:
            gamedf.loc[:, col] = gamedf.loc[:, col].fillna('0')
        for col in ['starttime_unix', "playtime_unix", 'rank', 'cards_init', 'num_init', 'label_uid', 'leftcards',
                    'leftcards_face']:
            gamedf.loc[:, col] = gamedf.loc[:, col].fillna(method='ffill')
        # 剩余的其他数值列: type【主要】, 剩余牌数，各牌型数量，其他选手牌数等
        # gamedf.loc[:, 'type'] = gamedf.loc[:, "type"].fillna(0)
        for col in gamedf.columns:
            gamedf.loc[:, col] = gamedf.loc[:, col].fillna(0)

        # 输出前项填充的结果gamedf
        # gamedf.to_excel(f"F:/aaa/gamedf_ffill.xlsx", index=False)

        gamedf = gamedf.sort_values(by=['cards_order', 'seat_order'], ascending=[True, True])  # 根据出牌顺序排序
        gamedf.reset_index(drop=True, inplace=True)

        # 对手出牌信息
        gamedf.loc[:, 'rival_leadcards_type'] = 0  # 出牌的类型
        gamedf.loc[:, 'rival_leadcards_cards'] = '0'  # 出牌的ID组
        gamedf.loc[:, 'rival_leadcards_num_show'] = '0'  # 出牌的牌面信息
        #  是否需要出炸，及出炸情况的标记
        # 是否需要出炸,1是（牌型压不住，剩余牌重组压不住）0否 （牌型压不住，剩余牌重组压的住） 2其他情况
        gamedf.loc[:, 'need_bomb'] = 2
        gamedf.loc[:, 'label_bomb'] = 0  # 是否出炸
        # 对手信息：出完牌后剩余的牌力值,位置，
        gamedf.loc[:, "rival_cards_value"] = 9999  # 可能为正好最后一手牌，因此令默认值为9999
        gamedf.loc[:, "rival_position"] = 1  # 对手位置 ，1 上家，0 下家   默认取上家，因为上家是主要情况
        gamedf.loc[:, 'rival_leftcards_nums'] = 28  # 出牌对手的剩余牌数量 , 28表示无需记录
        gamedf.loc[:, 'rival_leftcards_nums_pair'] = 28  # 出牌对手队友的剩余牌数量， 28表示无需记录
        # 取队友剩余牌
        gamedf.loc[:, "leftcards_nums_pair"] = gamedf.loc[:, 'leftcards_nums'].shift(-2).shift(4)
        gamedf.at[0, "leftcards_nums_pair"] = 27
        gamedf.at[1, "leftcards_nums_pair"] = 27
        gamedf.at[2, "leftcards_nums_pair"] = gamedf.at[0, 'leftcards_nums']
        gamedf.at[3, "leftcards_nums_pair"] = gamedf.at[1, 'leftcards_nums']

        # 输出增加统计列后的gamedf
        # gamedf.to_excel(f"F:/aaa/gamedf_nums_pair.xlsx", index=False)

        idx_length = gamedf.shape[0]
        gamedf.loc[:, 'type'] = gamedf.loc[:, 'type'].astype(int)

        # 取出牌对手、出牌对手队友的出牌信息，剩余牌信息
        # 局第一手
        gamedf = get_rival_info(gamedf, info_idx=0, plus_idx=1, game_start=True)
        if gamedf.at[1, 'type'] > 0:
            # 局第两手的情况
            gamedf = get_rival_info(gamedf, info_idx=1, plus_idx=1, game_start=True)
            if gamedf.at[2, 'type'] == 0 and gamedf.at[3, 'type'] == 0:
                # 第二位（下家）， 不出，不出 的情况
                gamedf = get_rival_info(gamedf, info_idx=1, plus_idx=3, game_start=True)
        for idx in range(2, idx_length-3):
            if gamedf.at[idx, 'type'] > 0:
                gamedf = get_rival_info(gamedf, info_idx=idx, plus_idx=1, game_start=False)
                if gamedf.at[idx + 1, 'type'] == 0 and gamedf.at[idx + 2, 'type'] == 0:
                    # 第二位(下家)， 不出，不出 的情况
                    gamedf = get_rival_info(gamedf, info_idx=idx, plus_idx=3, game_start=False)
        for idx in [idx_length-3, idx_length-2]:
            # 局倒数第三手，第二手的情况
            if gamedf.at[idx, 'type'] > 0:
                gamedf = get_rival_info(gamedf, info_idx=idx, plus_idx=1, game_start=False)

        # 出牌对手出牌的统计标记
        for idx in range(idx_length - 3):
            leadcard_type = gamedf.at[idx, 'type']  # 出牌类型
            if leadcard_type in [4096, 16384, 32768, 524288]:
                # 对手出炸弹
                gamedf = compare_bomb(gamedf, idx, 1)
                if gamedf.at[idx+1, 'type'] == 0 and gamedf.at[idx+2, 'type'] == 0:
                    # 第二位，不出，不出 的情况
                    gamedf = compare_bomb(gamedf, idx, 3)

            elif leadcard_type in [1, 2, 4]:
                # 对手出单张, 对子，三张
                gamedf = compare_cards(gamedf, type_card=leadcard_type, source_idx=idx, plus_idx=1, stage=True)
                if gamedf.at[idx + 1, 'type'] == 0 and gamedf.at[idx + 2, 'type'] == 0:
                    # 第二位队友(下家)的情况（不出，不出）
                    gamedf = compare_cards(gamedf, type_card=leadcard_type,  source_idx=idx, plus_idx=3, stage=True)
            elif leadcard_type in [64, 256, 512]:
                # 顺子，连对，钢板
                gamedf = compare_cards(gamedf, type_card=leadcard_type, source_idx=idx, plus_idx=1, stage=False)
                if gamedf.at[idx + 1, 'type'] == 0 and gamedf.at[idx + 2, 'type'] == 0:
                    # 第二位队友(下家)的情况（不出，不出）
                    gamedf = compare_cards(gamedf, type_card=leadcard_type, source_idx=idx, plus_idx=3, stage=False)
            elif leadcard_type in [32]:
                # 出三带二，上家
                gamedf = compare_leftcards_exclude(gamedf, source_idx=idx, plus_idx=1)
                if gamedf.at[idx + 1, 'type'] == 0 and gamedf.at[idx + 2, 'type'] == 0:
                    # 第二位队友（下家）的情况（不出，不出）
                    gamedf = compare_leftcards_exclude(gamedf, source_idx=idx, plus_idx=3)

        for idx in range(idx_length-3, idx_length-1):
            # 倒数第三手和倒数第二手的情况
            leadcard_type = gamedf.at[idx, 'type']
            if leadcard_type in [4096, 16384, 32768, 524288]:
                # 对手出炸弹
                gamedf = compare_bomb(gamedf, idx, 1)
            elif leadcard_type in [1, 2, 4]:
                # 对手出单张, 对子，三张
                gamedf = compare_cards(gamedf, type_card=leadcard_type, source_idx=idx, plus_idx=1, stage=True)
            elif leadcard_type in [64, 256, 512]:
                # 顺子，连对，钢板
                gamedf = compare_cards(gamedf, type_card=leadcard_type, source_idx=idx, plus_idx=1, stage=False)
            elif gamedf.at[idx, 'type'] in [32]:
                # 出三带二
                gamedf = compare_leftcards_exclude(gamedf, source_idx=idx, plus_idx=1)

        gamedf_cols = ['startguid', 'uid', 'cards_order', "leftcards_nums_pair",
                       'rival_leftcards_nums', 'rival_leftcards_nums_pair',
                       "rival_cards_value", "rival_position", 'rival_leadcards_type', 'rival_leadcards_cards',
                       'rival_leadcards_num_show', 'need_bomb', 'label_bomb', ]

        gamedf = gamedf.loc[:, gamedf_cols]
        # gamedf.drop(columns=['seat_order', 'leftcards_nums', 'startguid', 'uid', ], inplace=True)
        statistic_df = statistic_df.append(gamedf, sort=False, ignore_index=True)  # concat会匹配index,ignore_index
        # 输出所有统计列标记完成后的gamedf == statistic_df
        # statistic_df.to_excel(f"F:/aaa/gamedf_statistic.xlsx", index=False)

    # df = pd.concat([df, statistic_df], axis=1) # 按顺序匹配总是可能存在index的问题
    # dd = pd.merge(df, statistic_df, on='playtime_unix', copy=False)  # 根据出牌时间来匹配队友剩余牌数
    # dd.to_excel(f"F:/aaa/gamedf_merge_playtime.xlsx", index=False)
    df = pd.merge(df, statistic_df, on=['startguid','uid','cards_order'], copy=False)  # 根据匹配队友剩余牌数
    # df.to_excel(f"F:/aaa/gamedf_merge_cards_order.xlsx", index=False)
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
        if int(row["rival_leadcards_type"]) == 1 and row["need_bomb"] == 1:
            return 1  # 单张不可压
        elif int(row["rival_leadcards_type"]) == 1 and row["need_bomb"] == 0:
            return 2  # 单张可压
        elif int(row["rival_leadcards_type"]) >= 4096:
            return 3  # 出炸
        elif int(row["rival_leadcards_type"]) == 512:
            return 4  # "钢板"
        elif int(row["rival_leadcards_type"]) in [256, 64]:
            return 5  # "连对or顺子"
        elif (int(row["rival_leadcards_type"]) in [2, 4, 32]) and (row["need_bomb"] == 0):
            return 6  # 对子，三张，三带二 可压
        elif (int(row["rival_leadcards_type"]) in [2, 4, 32]) and (row["need_bomb"] == 1):
            return 7  # 对子，三张，三带二 不可压
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

    # 处理出牌对手位置
    rival_position_dict = {'1': '上家', '0': '下家'}
    df.loc[:, "rival_position"] = df.loc[:, "rival_position"].astype(str)
    df.loc[:, "rival_position"] = df.loc[:, "rival_position"].map(rival_position_dict)

    # 筛选数据进行统计
    used_cols = ["startguid", 'rival_leftcards_nums', 'rival_leftcards_nums_pair', 'leftcards_nums_pair',
                 'leftbomb_nums', 'Nonebomb_lefthands', 'lead_cards', 'rival_position', 'need_bomb', 'label_bomb']
    statistic_df = df.loc[
        (df.loc[:, 'need_bomb'] < 2) & (df.loc[:, 'lead_cards'] > 0) & (df.loc[:, 'label_uid'] > 0),
        used_cols]
    statistic_df.reset_index(drop=True, inplace=True)
    data_length = statistic_df.startguid.nunique()
    if statistic_df.shape[0]:
        statistic_df.loc[:, 'need_bomb'] = statistic_df.loc[:, 'need_bomb'].astype(int)
        statistic_df.loc[:, 'label_bomb'] = statistic_df.loc[:, 'label_bomb'].astype(int)
        statistic_df = statistic_df.groupby(used_cols[:-2]).agg({
            'need_bomb': pd.Series.count,
            'label_bomb': np.sum,
        })
        statistic_df.reset_index(drop=False, inplace=True)
        statistic_df.loc[:, "lead_cards"] = statistic_df.loc[:, 'lead_cards'].astype(str)
        lead_cards_name_dict = get_config_dict(os.getcwd(), 'robot_bomb_colnames.cfg', 'lead_cards_name_dict')
        statistic_df.loc[:, "lead_cards"] = statistic_df.loc[:, 'lead_cards'].map(lead_cards_name_dict)

        robot_bomb_statistic_situations = {
            "Nonebomb_lefthands": "手牌非炸手数",
            "Nonebomb_MeanCardvalue": "手牌非炸平均牌值",
            "lead_cards": "对手出牌",
            "leftbomb_nums": "自己剩余炸弹数",
            "leftcards_nums_pair": "队友剩余手牌数",
            "rival_leftcards_nums_pair": "出牌对手队友剩余手牌数",
            "rival_leftcards_nums": "出牌对手剩余手牌数",
            'rival_position': "出牌对手位置",
            "need_bomb": "occurrence_times",
            "label_bomb": "lead_bomb_times",
            'startguid': "房间号",
        }
        statistic_df.rename(columns=robot_bomb_statistic_situations, inplace=True)

        statistic_df.loc[:, 'probability'] = statistic_df.apply(
            lambda row: round(row["lead_bomb_times"] / row["occurrence_times"], 4)
            if row["occurrence_times"] != 0 else 0, axis=1)

        # 存储统计结果
        write_data(statistic_df, filedir=os.path.join(tmpdatadir1, "20190709"), filename='statistic_result', )
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
        # get_raw_test_data(True,
        #                   ["81_10163_0_1562633295_4",
        #                    '81_10163_0_1562640783_4',
        #                    '81_8136_0_1562602103_4',
        #                    '81_8136_0_1562602871_4',
        #                    '81_18148_0_1562603113_4',
        #                    '81_18148_0_1562605573_4',
        #                    '81_19015_0_1562602841_4',
        #                    '81_21040_0_1562603449_4'])  # 生成测试数据
        mergedata = get_raw_test_data()  # 读取测试数据

    else:
        mergedata = get_raw_data(win_ratio=win_ratio)  # 读取正式数据

    def combine_result_files(result_filepath):
        if os.path.exists(result_filepath):
            #  标记结果合并
            robot_result = pd.DataFrame()
            current_result_dir = os.path.join(tmpdatadir1, "20190709")
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
        for start_index in range(0, sep_bins_length-1):
            chunk_df = mergedata.loc[
                mergedata.loc[:, 'startguid'].isin(unique_startguid[sep_bins[start_index]:sep_bins[start_index + 1]])]
            chunk_df.reset_index(drop=True, inplace=True)
            chunk_df = basic_treatment(chunk_df)  # 初步处理
            chunk_df = calculate_cards_value(chunk_df)  # 拆牌结果
            # write_data(chunk_df, filename='aaaaaaaaa')
            chunk_df, data_length = statistic_procedure_v2(chunk_df)  # 前置，统计结果
            if chunk_df.shape[0]:
                # 生成latest版本统计结果
                circle_statistic_procedure(chunk_df, os.path.join(tmpdatadir1, '20190709'),
                                           data_length=data_length, all_data_length=sep_bins[start_index + 1])
            if start_index % 1000 == 0:
                combine_result_files(os.path.join(tmpdatadir1, '20190709', 'detail_result'))
            if start_index == sep_bins_length-1:
                combine_result_files(os.path.join(tmpdatadir1, '20190709', 'detail_result'))
            # del chunk_df

    # 后置，统计数据提取
    # do_statistic_procedure(tmpdatadir1, prefix='robot_result', grouper=False, out_name="statistic_result")


if __name__ == '__main__':
    reduce_raw_data()  # 缩减原始数据体积
    # main_process(True, data_sep=1)  # 测试数据
    main_process(process_test=False, win_ratio=0.5, data_sep=1, )
