# -*- encoding: utf-8 -*-
# @Version : 1.0
# @Time    : 2019/6/5 14:05
# @Author  : wanghd
# @note    : 用户牌局出炸情况处理（拆牌+ 统计）


import os
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


def write_data(df, filedir='D:/projectsHome/ur.ctwl/tmpdata1', filename='robot_bomb', index=False):
    current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
    filename = f'{filename}-{current_time}.xlsx'
    df.to_excel(os.path.join(filedir, filename), index=index)


def get_partialdata():
    """对原始数据仅提取需要的字段"""
    # 对局id,uid, 开局时间，每手出牌时间点，出牌状态（出，不出，托管）'optype',
    # 牌型type，出牌回合, 出牌信息id[0-107]，出牌数字信息[1,14]
    # showcards_usecols = ['startguid', 'uid', 'starttime_unix', 'playtime_unix', 'type', 'cards_order', 'cards', 'num']
    # 对局id，uid, 起手牌牌id[0-107], 起手牌牌数字[1-14], 级牌
    cus_usecols = ['startguid', 'uid', 'cards', 'num', 'rank']
    data = pd.read_csv(os.path.join(rawdatadir, 'customization_20190527.csv'), usecols=cus_usecols)
    data.to_excel(os.path.join(rawdatadir1, 'customization527.xlsx'))


def get_raw_test_data():
    """测试数据"""
    # cus_usecols = ['startguid', 'uid', 'cards', 'num','rank'] 起手牌牌id[0-107], 起手牌牌数字[1-14]，级牌
    cus = pd.read_excel(os.path.join(rawdatadir1, 'cus_test.xlsx'))
    name_dict = {'cards': "cards_init", 'num': 'num_init'}
    cus.rename(columns=name_dict, inplace=True)

    # 对局id,uid, 开局时间，每手出牌时间点，出牌状态（出，不出，托管）'optype',
    # 牌型type，出牌回合, 出牌信息id[0-107]，出牌数字信息[1,14]
    # ['startguid', 'uid', 'starttime_unix', 'playtime_unix', 'type', 'cards_order', 'cards', 'num']
    show = pd.read_excel(os.path.join(rawdatadir1, 'show_test.xlsx'))

    cus['cards_order'] = 1
    mergedata = pd.merge(show, cus, on=['startguid', 'uid', 'cards_order'], how='left', copy=False)
    return mergedata


def get_raw_data():
    """读取正式数据"""
    # cus_usecols = ['startguid', 'uid', 'cards', 'num','rank'] 起手牌牌id[0-107], 起手牌牌数字[1-14]
    cus_usecols = ['startguid', 'uid', 'cards', 'rank']  # 起手牌，级牌
    cus_files = [file for file in os.listdir(rawdatadir) if file.startswith('short_customization_20190527')]
    print(cus_files)
    name_dict = {'cards': "cards_init", 'num': 'num_init'}  # 重命名为 初始牌组

    # 对局id,uid, 开局时间，每手出牌时间点，出牌状态（出，不出，托管）'optype',
    # 牌型type，出牌回合, 出牌信息id[0-107]，出牌数字信息[1,14]
    # ['startguid', 'uid', 'starttime_unix', 'playtime_unix', 'type', 'cards_order', 'cards', 'num']
    show_usecols = ['startguid', 'uid', 'starttime_unix', 'playtime_unix', 'type', 'cards_order', 'cards']
    show_files = [file for file in os.listdir(rawdatadir) if file.startswith('short_showcards_20190527')]
    print(show_files)
    for cus_file, show_file in zip(cus_files, show_files):
        cus = pd.read_csv(os.path.join(rawdatadir, cus_file), usecols=cus_usecols)  # 读取定制信息
        cus.rename(columns=name_dict, inplace=True)  # 重命名为初始牌组
        cus['cards_order'] = 1  # 起手牌标记出牌回合为 1
        show = pd.read_csv(os.path.join(rawdatadir, show_file), usecols=show_usecols)  # 读取出牌信息

        # 将初始牌组 和 出牌信息合并
        show = pd.merge(show, cus, on=['startguid', 'uid', 'cards_order'], how='left', copy=False)
        del cus
        return show


def reduce_rawdata():
    """缩减原始数据体积，仅保存需要的列"""
    first_init()
    # 缩减customization 定制表体积
    cus_usecols = ['startguid', 'uid', 'cards', 'rank']  # 起手牌,级牌
    cus_files = [file for file in os.listdir(rawdatadir) if file.startswith('customization')]
    cus_files = cus_files[-4:]
    print(cus_files)
    for cus_file in cus_files:
        cus = pd.read_csv(os.path.join(rawdatadir, cus_file), usecols=cus_usecols)
        cus.to_csv(os.path.join(rawdatadir, f"short_{cus_file}"), index=False)

    # 缩减showcards 出牌信息表体积
    show_usecols = ['startguid', 'uid', 'starttime_unix', 'playtime_unix', 'type', 'cards_order', 'cards']
    show_files = [file for file in os.listdir(rawdatadir) if file.startswith('showcards')]
    show_files = show_files[-4:]
    print(show_files)
    for show_file in show_files:
        show = pd.read_csv(os.path.join(rawdatadir, show_file), usecols=show_usecols)
        show.to_csv(os.path.join(rawdatadir, f"short_{show_file}"), index=False)


def basic_treatment(mergedata):
    """
    对于原始局内信息做初步处理：级牌 、 构建累积出牌、 初始牌 、 剩余牌 、
    :param mergedata:
    :return:
    """
    # 填充对局的rank
    mergedata.loc[:, 'rank'] = mergedata.loc[:, 'rank'].fillna(method='ffill')

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
    inputfile = os.path.join(base_path, 'Input.txt')
    if os.path.exists(inputfile):
        os.remove(inputfile)
    outputfile = os.path.join(base_path, 'Output.txt')
    if os.path.exists(outputfile):
        os.remove(outputfile)

    df_tmp = df.loc[df.loc[:, 'type'] > 0, ['startguid', 'uid', 'cards_order', 'rank', 'cards']]
    df_tmp = df_tmp.reset_index(drop=True)
    df_tmp.loc[:, 'leadcards_str'] = df_tmp.apply(
        lambda row: str(row["startguid"]) + str(row["uid"]) + str(row["cards_order"]), axis=1)
    # 写入牌面信息 到 input
    with open(os.path.join(base_path, 'Input.txt'), 'a') as f:
        for rowid in range(df_tmp.shape[0]):
            f.write(df_tmp.at[rowid, 'leadcards_str'])
            card_list = str(df_tmp.at[rowid, 'cards']).split(',')
            basic_list = ['-1'] * (33 - len(card_list))
            card_list.extend(basic_list)
            f.write(os.linesep)
            # 将级牌拼接到牌组后面
            cards_str = ','.join([str(cards_id) for cards_id in card_list])
            current_rank = str(int(df_tmp.at[rowid, 'rank']))
            cards_str = f'{cards_str}||{current_rank}'
            f.write(cards_str)
            f.write(os.linesep)

    # del df_tmp  # 删除仅包含出牌的df_tmp
    # 执行 CalculatorCardsValue.exe 获得拆牌信息
    subprocess.call(os.path.join(base_path, 'CalculatorCardsValue3.exe'))

    # 读取拆牌信息
    df_cardsvalue = pd.DataFrame(columns=['leadcards_str', 'leadcards_cardvalue'])
    with open(os.path.join(base_path, 'Output.txt'), 'r') as fout:
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
    inputfile = os.path.join(base_path, 'Input.txt')
    if os.path.exists(inputfile):
        os.remove(inputfile)
    outputfile = os.path.join(base_path, 'Output.txt')
    if os.path.exists(outputfile):
        os.remove(outputfile)
    # 写入牌面信息 到 input
    with open(os.path.join(base_path, 'Input.txt'), 'a') as f:
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
                current_rank = str(int(df.at[rowid, 'rank']))
                cards_str = f'{cards_str}||{current_rank}'
                f.write(cards_str)
                f.write(os.linesep)
                previous_card = df.at[rowid, 'leftcards_str']

    # 执行 CalculatorCardsValue.exe 获得拆牌信息
    subprocess.call(os.path.join(base_path, 'CalculatorCardsValue3.exe'))

    # win32api.keybd_event(13, 0, 0, 0)  # enter 键 位码是13  按键
    # win32api.keybd_event(17, 0, 0, 0)  # ctrl 键 位码是17
    # win32api.keybd_event(13, 0, win32con.KEYEVENTF_KEYUP, 0) 释放按键
    # win32api.keybd_event(17, 0, win32con.KEYEVENTF_KEYUP, 0)

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
            # print(index, line)

    df_cardsvalue.drop_duplicates(inplace=True)
    # 存储拆牌前的数据
    # write_data(df, filename='robot_001_notchaipai')

    df = pd.merge(df, df_cardsvalue, on=['leftcards_str'], how='left', copy=False)
    df.drop(columns=["leftcards_str"], inplace=True)  # 删除合并标识列 leftcards_str

    df = calculate_lead_cards_value(df)  # 出牌牌值计算
    df = apart_cards_type(df)  # 牌型拆解
    df = rival_leadcards_treatment(df)  # 提取对手出牌，各出牌情况标记，对手剩余牌
    statistic_procedure(df)  # 统计数据提取
    read_data_result(os.path.join(tmpdatadir1, 'statistic.txt'))
    # 存储拆牌后的数据
    write_data(df, filename='robot_result')
    return df


def apart_cards_type(df):
    """
    牌型拆解,
    :param df: 经过牌力值程序计算之后带有 牌型 的 dataframe
    :return: 各牌型数量，剩余手牌中各牌型的最大牌值，
    """

    # 拆分出各牌型的数量
    # df.loc[:, 'card_type_list'] = df.loc[:, 'cards_type'].apply(
    #     lambda x: [x.split(sep='-')[0] for x in str(x).split(sep='||')])  # 牌型组成的列表
    # df.loc[:, 'card_type_list_counter'] = df.loc[:, 'card_type_list'].apply(Counter)  # 计算牌型数量
    # 将counter 展开为列
    # df_cardnum = pd.DataFrame(df["card_type_list_counter"].values.tolist())
    # df = pd.concat([df, df_cardnum], axis=1)
    # cardtype_namedict = {'1': '单张', '2': '对子', '4': '三张', '64': '顺子', '256': '连对', '512': '钢板', '4096': '炸弹4-5张',
    #                      '16384': "同花顺", "32768": '超级大炸弹', '四王': '524288', }  # 3带2=='32'
    # df.rename(columns=cardtype_namedict, inplace=True)
    # if 'nan' in df.columns:
    #     # 对于出完牌的情况，Counter返回 nan 的计数，因此删除
    #     df.drop(columns=['nan'], inplace=True)
    # df.drop(columns=["card_type_list_counter", 'card_type_list'], inplace=True)  # 删除中间 Counter 结果,牌类型列表

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

    # df = pd.concat([df, df.loc[:,'apartcol'].str.split(',', expand=True)], axis=1, names=colnames_optypenumlist)

    # optype_counter = Counter(card_optype_list)
    # createVar = locals()
    # for optype in cardoptypelist:
    #     createVar['optype'+str(optype)+'_nums'] = optype_counter[str(optype)]

    # optype1     = optype_counter["1"]
    # optype2     = optype_counter["2"]
    # optype4     = optype_counter["4"]
    # optype64    = optype_counter["64"]
    # optype256   = optype_counter["256"]
    # optype4096  = optype_counter["4096"]
    # optype16384 = optype_counter["16384"]
    # optype32768 = optype_counter["32768"]
    # optype524288= optype_counter["524288"]

    # 统计手牌非炸弹牌型平均牌值
    def calculate_mean_cards_value(x):
        tmp_list = []
        cards_list = str(x).split(sep='||')
        # print(cards_list)
        for card in cards_list:
            if 'nan' not in card:
                card_type, _, card_value = card.split(sep='-')
                if card_type not in ['4096', '16384', '32768', '524288']:
                    tmp_list.append(int(card_value))
        return np.round(np.mean(tmp_list), 2)

    # 统计手牌非炸弹牌型平均牌值
    df.loc[:, 'Nonebomb_MeanCardvalue'] = df.loc[:, 'cards_type'].apply(calculate_mean_cards_value)

    # 手牌非炸弹手数( 可以与 统计平均牌值 合并为一个方法)
    def calculate_lefthands(x):
        """非炸弹手牌：出完牌的手数"""
        tmp_list = []
        cards_list = str(x).split(sep='||')  # ['1-1-8','2-2-10'],
        # print(cards_list)
        for card in cards_list:
            if 'nan' not in card:
                card_type, _, _ = card.split(sep='-')
                if card_type not in ['4096', '16384', '32768', '524288']:
                    tmp_list.append(card_type)
        # 所有牌型数量- min{对子数，三张数}
        shoushu = len(tmp_list) - min(tmp_list.count('2'), tmp_list.count("4"))
        return shoushu

    # 统计手牌非炸弹牌 手数——几手出完
    df.loc[:, 'Nonebomb_lefthands'] = df.loc[:, 'cards_type'].apply(calculate_lefthands)

    # 剩余手牌中炸弹数量
    def calculate_leftbombs(x):
        """手牌中炸弹数量"""
        tmp_list = []
        cards_list = str(x).split(sep='||')  # ['1-1-8','2-2-10'],
        # print(cards_list)
        for card in cards_list:
            if 'nan' not in card:
                card_type, _, _ = card.split(sep='-')
                if card_type in ['4096', '16384', '32768', '524288']:
                    tmp_list.append(card_type)
        return len(tmp_list)

    # 剩余炸弹数量
    df.loc[:, 'leftbomb_nums'] = df.loc[:, 'cards_type'].apply(calculate_leftbombs)

    # 剩余最大炸弹判断
    def calculate_maxbomb(x):
        """判断剩余最大炸弹的大小，【4炸小于10 --1，同花顺--2，其他--0】"""
        bomb_list = []  # 由炸弹 (cardtype, cardnum, cardvalue) 元组构成的列表
        cards_list = str(x).split(sep='||')  # ['1-1-8','2-2-10'],
        for card in cards_list:
            if 'nan' not in card:
                card_type, card_num, card_value = card.split(sep='-')
                if card_type in ['4096', '16384', '32768', '524288']:
                    bomb_list.append((card_type, card_num, card_value))

        bomb_list_length = len(bomb_list)
        if bomb_list_length > 0:
            card_type, card_num, card_value = bomb_list[-1]
            if card_type in ['16384', '32768', '524288']:
                return 2
            elif card_type in ['4096'] and card_num in ['4'] and int(card_value) < 9:
                return 1
            else:
                return 0
        else:
            return 0

    # 判断 剩余的最大炸弹
    df.loc[:, 'leftmaxbomb'] = df.loc[:, 'cards_type'].apply(calculate_maxbomb)

    # write_data(df, tmpdatadir1, filename='robot222')
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
                inhand_bomb_max_cardvalue = int(np.max(gamedf.loc[
                                                           idx + 1, ['optype4096_maxcardvalue',
                                                                     'optype16384_maxcardvalue',
                                                                     'optype32768_maxcardvalue',
                                                                     'optype524288_maxcardvalue']].values))
                if inhand_bomb_max_cardvalue > int(gamedf.at[idx, 'leadcards_cardvalue']):
                    gamedf.at[idx + 1, 'lead_bomb'] = 1  # 标记对手出炸
                    gamedf.at[idx + 1, 'need_bomb'] = 1  # 需要出炸
                    if gamedf.at[idx, "first_hand"] > 0:
                        gamedf.at[idx + 1, 'lead_firsthand_bomb'] = 1  # 标记为开局出炸
                    if gamedf.at[idx + 1, 'type'] > 0:
                        gamedf.at[idx + 1, 'label_bomb'] = 1  # 出炸

                if gamedf.at[idx+1, 'type'] == 0 and gamedf.at[idx+2, 'type'] == 0:
                    # 第二位，不出，不出 的情况
                    inhand_bomb_max_cardvalue = int(np.max(gamedf.loc[
                                                               idx + 1, ['optype4096_maxcardvalue',
                                                                         'optype16384_maxcardvalue',
                                                                         'optype32768_maxcardvalue',
                                                                         'optype524288_maxcardvalue']].values))
                    if inhand_bomb_max_cardvalue > int(gamedf.at[idx, 'leadcards_cardvalue']):
                        gamedf.at[idx + 3, 'lead_bomb'] = 1  # 标记对手出炸
                        gamedf.at[idx + 3, 'need_bomb'] = 1  # 需要出炸
                        if gamedf.at[idx, "first_hand"] > 0:
                            gamedf.at[idx + 3, 'lead_firsthand_bomb'] = 1  # 标记第二个玩家的开局出炸
                        if gamedf.at[idx + 3, 'type'] > 0:
                            gamedf.at[idx + 3, 'label_bomb'] = 1  # 出炸
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
                if int(max(gamedf.loc[
                               idx + 1, ['optype4096_maxcardvalue', 'optype16384_maxcardvalue',
                                         'optype32768_maxcardvalue',
                                         'optype524288_maxcardvalue']])) > int(gamedf.at[idx, 'leadcards_cardvalue']):
                    gamedf.at[idx + 1, 'lead_bomb'] = 1  # 标记对手出炸
                    gamedf.at[idx + 1, 'need_bomb'] = 1  # 需要出炸
                    if gamedf.at[idx + 1, 'type'] > 0:
                        gamedf.at[idx + 1, 'label_bomb'] = 1  # 出炸
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


def statistic_procedure(df):
    """提取各情况的统计值"""
    def get_write_data(label_name, col_name, col_value_list):
        # 获得统计值的辅助函数
        df.loc[:, col_name] = df.loc[:, col_name].astype(float)
        if len(col_value_list) > 1:
            tmp_df = df.loc[
                (df.loc[:, col_name] > col_value_list[0]) & (df.loc[:, col_name] < col_value_list[1])]
        else:
            tmp_df = df.loc[df.loc[:, col_name] == col_value_list[0]]
        tmp_df = tmp_df.loc[tmp_df.loc[:, 'need_bomb'] == 1]
        need_bomb_times = sum(tmp_df.loc[:, 'need_bomb'])
        label_bomb_times = sum(tmp_df.loc[:, 'label_bomb'])
        format_str = f'{label_name}, {str(need_bomb_times)}, {str(label_bomb_times)}'
        return format_str

    # current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
    filename = 'statistic.txt'
    with open(os.path.join(tmpdatadir1, filename), 'a') as f:
        # 出牌对手剩余手牌数
        f.write(get_write_data('rival_leftcards_numsgt15', 'rival_leftcards_nums', [15, 28]))
        f.write(os.linesep)
        f.write(get_write_data('rival_leftcards_numsgt10lt15', 'rival_leftcards_nums', [10, 15]))
        f.write(os.linesep)
        f.write(get_write_data('rival_leftcards_numsgt7lt10', 'rival_leftcards_nums', [7, 10]))
        f.write(os.linesep)
        f.write(get_write_data('rival_leftcards_numsgt4lt8', 'rival_leftcards_nums', [4, 8]))
        f.write(os.linesep)
        f.write(get_write_data('rival_leftcards_numseq4', 'rival_leftcards_nums', [4]))
        f.write(os.linesep)
        f.write(get_write_data('rival_leftcards_numseq3', 'rival_leftcards_nums', [3]))
        f.write(os.linesep)
        f.write(get_write_data('rival_leftcards_numseq2', 'rival_leftcards_nums', [2]))
        f.write(os.linesep)
        f.write(get_write_data('rival_leftcards_numseq1', 'rival_leftcards_nums', [1]))
        f.write(os.linesep)
        f.write(get_write_data('rival_leftcards_numseq0', 'rival_leftcards_nums', [0]))
        f.write(os.linesep)
        # 出牌对手队友剩余手牌数
        f.write(get_write_data('rival_leftcards_nums_pairgt15', 'rival_leftcards_nums_pair', [15, 28]))
        f.write(os.linesep)
        f.write(get_write_data('rival_leftcards_nums_pairgt10lt15', 'rival_leftcards_nums_pair', [10, 15]))
        f.write(os.linesep)
        f.write(get_write_data('rival_leftcards_nums_pairgt7lt10', 'rival_leftcards_nums_pair', [7, 10]))
        f.write(os.linesep)
        f.write(get_write_data('rival_leftcards_nums_pairgt4lt8', 'rival_leftcards_nums_pair', [4, 8]))
        f.write(os.linesep)
        f.write(get_write_data('rival_leftcards_nums_paireq4', 'rival_leftcards_nums_pair', [4]))
        f.write(os.linesep)
        f.write(get_write_data('rival_leftcards_nums_paireq3', 'rival_leftcards_nums_pair', [3]))
        f.write(os.linesep)
        f.write(get_write_data('rival_leftcards_nums_paireq2', 'rival_leftcards_nums_pair', [2]))
        f.write(os.linesep)
        f.write(get_write_data('rival_leftcards_nums_paireq1', 'rival_leftcards_nums_pair', [1]))
        f.write(os.linesep)
        f.write(get_write_data('rival_leftcards_nums_paireq0', 'rival_leftcards_nums_pair', [0]))
        f.write(os.linesep)
        # 队友剩余手牌数
        f.write(get_write_data('leftcards_nums_pairgt15', 'leftcards_nums_pair', [15, 28]))
        f.write(os.linesep)
        f.write(get_write_data('leftcards_nums_pairgt10lt15', 'leftcards_nums_pair', [10, 15]))
        f.write(os.linesep)
        f.write(get_write_data('leftcards_nums_pairgt7lt10', 'leftcards_nums_pair', [7, 10]))
        f.write(os.linesep)
        f.write(get_write_data('leftcards_nums_pairgt4lt8', 'leftcards_nums_pair', [4, 8]))
        f.write(os.linesep)
        f.write(get_write_data('leftcards_nums_paireq4', 'leftcards_nums_pair', [4]))
        f.write(os.linesep)
        f.write(get_write_data('leftcards_nums_paireq3', 'leftcards_nums_pair', [3]))
        f.write(os.linesep)
        f.write(get_write_data('leftcards_nums_paireq2', 'leftcards_nums_pair', [2]))
        f.write(os.linesep)
        f.write(get_write_data('leftcards_nums_paireq1', 'leftcards_nums_pair', [1]))
        f.write(os.linesep)
        f.write(get_write_data('leftcards_nums_paireq0', 'leftcards_nums_pair', [0]))
        f.write(os.linesep)
        # 自己剩余牌信息
        f.write(get_write_data('leftbomb_numseq1', 'leftbomb_nums', [1]))
        f.write(os.linesep)
        f.write(get_write_data('leftbomb_numseq2', 'leftbomb_nums', [2]))
        f.write(os.linesep)
        f.write(get_write_data('leftbomb_numseq3', 'leftbomb_nums', [3]))
        f.write(os.linesep)
        f.write(get_write_data('leftbomb_numseq4', 'leftbomb_nums', [4]))
        f.write(os.linesep)
        f.write(get_write_data('leftbomb_numsgt4', 'leftbomb_nums', [4, 8]))
        f.write(os.linesep)
        f.write(get_write_data('leftmaxbombeq2', 'leftmaxbomb', [2]))  # 同花顺及以上
        f.write(os.linesep)
        f.write(get_write_data('leftmaxbombeq1', 'leftmaxbomb', [1]))  # 4炸小于10
        f.write(os.linesep)
        # 手牌非炸弹手数
        f.write(get_write_data('Nonebomb_lefthandseq1', 'Nonebomb_lefthands', [1]))
        f.write(os.linesep)
        f.write(get_write_data('Nonebomb_lefthandseq2', 'Nonebomb_lefthands', [2]))
        f.write(os.linesep)
        f.write(get_write_data('Nonebomb_lefthandseq3', 'Nonebomb_lefthands', [3]))
        f.write(os.linesep)
        f.write(get_write_data('Nonebomb_lefthandseq4', 'Nonebomb_lefthands', [4]))
        f.write(os.linesep)
        f.write(get_write_data('Nonebomb_lefthandsgt4', 'Nonebomb_lefthands', [4, 20]))  # 手牌非炸手数大于4
        f.write(os.linesep)
        # 手牌非炸平均牌值
        f.write(get_write_data('Nonebomb_lefthandslteq8', 'Nonebomb_lefthands', [0, 8.01]))  # (:,8]
        f.write(os.linesep)
        f.write(get_write_data('Nonebomb_lefthandsgt8lteq11', 'Nonebomb_lefthands', [8, 11.01]))  # （8,11]
        f.write(os.linesep)
        f.write(get_write_data('Nonebomb_lefthandsgt11', 'Nonebomb_lefthands', [11, 50]))  # (11,:)
        f.write(os.linesep)
        # 对手出牌情况
        f.write(get_write_data('lead_firsthand_bombeq1', 'lead_firsthand_bomb', [1]))  # 开局炸
        f.write(os.linesep)
        f.write(get_write_data('lead_bombeq1', 'lead_bomb', [1]))  # 出炸
        f.write(os.linesep)
        f.write(get_write_data('lead_max_optype1eq1', 'lead_max_optype1', [1]))  # 最大单张
        f.write(os.linesep)
        f.write(get_write_data('lead_max_optype2eq1', 'lead_max_optype2', [1]))  # 最大对子
        f.write(os.linesep)
        f.write(get_write_data('lead_max_optype4eq1', 'lead_max_optype4', [1]))  # 最大三张
        f.write(os.linesep)
        f.write(get_write_data('lead_max_optype32eq1', 'lead_max_optype32', [1]))  # 最大三带二
        f.write(os.linesep)
        f.write(get_write_data('lead_max_optype256eq1', 'lead_max_optype256', [1]))  # 连对
        f.write(os.linesep)
        f.write(get_write_data('lead_max_optype512eq1', 'lead_max_optype512', [1]))  # 钢板
        f.write(os.linesep)
        f.write(get_write_data('lead_max_optype64eq1', 'lead_max_optype64', [1]))  # 顺子
        f.write(os.linesep)
        f.write(get_write_data('optype12432eq1', 'optype12432', [1]))  # 单张、对子、三张、三带二非炸压不住
        f.write(os.linesep)
        f.write(get_write_data('optype25651264eq1', 'optype25651264', [1]))  # 连对、钢板、顺子非炸压不住
        f.write(os.linesep)


def read_data_result(filepath):
    basename = os.path.dirname(filepath)
    filename = str(os.path.basename(filepath)).split(".")[0]
    with open(filepath, 'r') as f:
        data = [line.strip().split(',') for line in f.readlines()]
    data = pd.DataFrame(data, columns=['situation', 'occurrence_times', 'lead_bomb_times'])  # 情景，出现次数，出炸次数
    data.dropna(inplace=True)  # 删除换行
    data.loc[:, 'occurrence_times'] = data.loc[:, 'occurrence_times'].astype(int)
    data.loc[:, 'lead_bomb_times'] = data.loc[:, 'lead_bomb_times'].astype(int)
    data = data.groupby(['situation']).agg({
        'occurrence_times': np.sum,
        'lead_bomb_times': np.sum,
    })
    data.loc[:, 'probability'] = data.apply(
        lambda row: round(row["lead_bomb_times"] / row["occurrence_times"], 4) if row["occurrence_times"] != 0 else 0,
        axis=1)
    write_data(data, basename, filename, index=True)


def main_process(process_test=True, data_sep=10000):
    first_init()   # 初始化配置
    if process_test:
        mergedata = get_raw_test_data()  # 读取测试数据
    else:
        mergedata = get_raw_data()  # 读取正式数据

    unique_startguid = mergedata.loc[:, "startguid"].unique()
    unique_startguid_length = len(unique_startguid)
    if unique_startguid_length < data_sep:
        mergedata = basic_treatment(mergedata)  # 初步处理
        calculate_cards_value(mergedata)  # 拆牌, 统计
    else:
        sep_bins = list(range(0, unique_startguid_length+data_sep, data_sep))
        for start_index in range(len(sep_bins)-1):
            chunk_df = mergedata.loc[
                mergedata.loc[:, 'startguid'].isin(unique_startguid[sep_bins[start_index]:sep_bins[start_index + 1]])]
            chunk_df = basic_treatment(chunk_df)  # 初步处理
            calculate_cards_value(chunk_df)  # 拆牌, 统计
            del chunk_df


if __name__ == '__main__':
    # reduce_rawdata()
    # main_process(True)  # 测试数据
    # main_process(process_test=False, data_sep=3000)

    first_init()
    allfiles = [file for file in os.listdir(tmpdatadir1) if file.startswith("robot_result")]
    df = pd.DataFrame()
    for file in allfiles:
        tmp_df = pd.read_excel(os.path.join(tmpdatadir1, file))
        df = df.append(tmp_df, sort=False)
    statistic_procedure(df)  # 统计数据提取
    read_data_result(os.path.join(tmpdatadir1, 'statistic.txt'))
