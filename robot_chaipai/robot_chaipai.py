# -*- encoding: utf-8 -*-
# @Version : 2.0
# @Time    : 2019/7/3 17:07
# @Time    : 2019/8/6 16:00 多目录同时处理
# @Time    : 2019/8/14 拆牌来源按牌ID在上一手牌中的牌型判定
# @Author  : wanghd
# @note    : v1 拆牌来源情况统计：依赖于【出炸统计】（robot_result）的标记结果文件； 仅针对单，对子，三张
# @note    : v2 拆牌来源情况统计：依赖于【出炸统计】（robot_result）的标记结果文件；牌ID在上一手牌的牌组判定为来源

import os
import re
import time
import random
import pandas as pd
from tqdm import tqdm


def first_init():
    """初始化基础配置"""
    global tmpdatadir1

    # tmpdatadir1 = os.path.abspath(r'F:\chuzha_chaipai_results')
    tmpdatadir1 = os.path.abspath(r'D:\projectsHome\ur.ctwl\tmpdata1')


def write_data(df, filedir=os.getcwd(), filename='chaipai_result', index=False):
    if not os.path.exists(filedir):
        os.mkdir(filedir)
    current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
    filename = f'{filename}_{current_time}_{random.randint(1, 10)}.csv'
    df.to_csv(os.path.join(filedir, filename), index=index, header=True, encoding='gbk')


def chaipai_apart(detail_result_dir):
    """只适用于单张，对子，三张的拆牌,6.27号数据"""
    cardoptypelist = [1, 2, 4, 64, 256, 512, 4096, 16384, 32768, 524288]
    colnames_optypenumlist = [f"optype{optype}_nums" for optype in cardoptypelist]
    # result_dir = os.path.join(tmpdatadir1, 'detail_result')
    result_dir = detail_result_dir
    # robot_result 结果文件目录
    robot_result_files = [file for file in os.listdir(result_dir) if file.startswith('robot_result')]
    for file in robot_result_files:
        data = pd.read_csv(os.path.join(result_dir, file), encoding='gbk')
        data.loc[:, 'type'] = data.loc[:, 'type'].astype(int)
        # data = data.loc[data.loc[:, 'type'].isin([1, 2, 4])].reset_index(drop=True)
        start_guids = data.loc[:, 'startguid'].unique()

        col_names = list(data.columns)  # 生成结果文件的列名
        col_names = col_names.extend(["mad", "source"])  # 生成结果文件的列名
        chaipai_result = pd.DataFrame(columns=col_names)

        for start_guid in start_guids:
            gamedf = data.loc[data.loc[:, 'startguid'] == start_guid]
            uids = gamedf.uid.unique()
            for uid in uids:
                user_df = gamedf.loc[gamedf.loc[:, 'uid'] == uid]
                user_df = user_df.sort_values(by=["playtime_unix"], ascending=True)  # 根据出牌顺序排序
                user_df.reset_index(drop=True, inplace=True)

                dd = user_df.loc[:, colnames_optypenumlist].diff()
                dd.fillna(0, inplace=True)
                # 计算1,2,4，单张，对子，三张的变化情况
                dd.loc[:, "mad"] = dd.apply(lambda row: sum([abs(row[x]) for x in colnames_optypenumlist[:3]]), axis=1)

                def find_souce(row):
                    value_list = list(row.values.tolist())
                    value_list.reverse()
                    if -1 in value_list:
                        return cardoptypelist[len(value_list) - 1 - value_list.index(-1)]
                    else:
                        return 0

                dd.loc[:, "source"] = dd.apply(find_souce, axis=1)
                dd = pd.concat([user_df, dd], axis=1, sort=False)

                # 筛选产生拆牌情况的数据，
                dd = dd.loc[dd.loc[:, 'mad'] != 1].reset_index(drop=True)
                # 仅保留需要的拆牌情况
                dd = dd.loc[dd.loc[:, 'type'].isin([1, 2, 4])].reset_index(drop=True)
                # 添加结果
                chaipai_result = chaipai_result.append(dd)

        if chaipai_result.shape[0]:
            write_data(chaipai_result, filedir=result_dir, filename='chaipai_result', index=False)


def chaipai_apart_new(detail_result_dir):
    """基于牌ID在上一手的牌型组，判断为其来源"""
    cardoptypelist = [1, 2, 4, 64, 256, 512, 4096, 16384, 32768, 524288]
    # colnames_optypenumlist = [f"optype{optype}_nums" for optype in cardoptypelist]
    result_dir = detail_result_dir
    # robot_result 结果文件目录
    robot_result_files = [file for file in os.listdir(result_dir) if file.startswith('robot_result')]
    for file in tqdm(robot_result_files):
        basic_cols = ["startguid", 'uid', 'playtime_unix', "type", 'label_uid', "num_show", "cards", 'rank',
                      "leftcards_nums", 'cards_order', 'cards_id', "cards_type"]
        data = pd.read_csv(os.path.join(result_dir, file), encoding='gbk', usecols=basic_cols)
        data.loc[:, 'type'] = data.loc[:, 'type'].astype(int)
        # data = data.loc[data.loc[:, 'type'].isin([1, 2, 4])].reset_index(drop=True)
        # 构建上一手出牌的信息
        used_cols = ["startguid", 'uid', 'cards_order', 'cards_id', "cards_type"]
        data_last_round = data.loc[:, used_cols].copy()
        new_cols = ["startguid", 'uid', 'cards_order', 'last_cards_id', "last_cards_type"]
        data_last_round.columns = new_cols
        data_last_round.loc[:, 'cards_order'] = data_last_round.loc[:, 'cards_order'] + 1
        data = pd.merge(data, data_last_round, on=["startguid", 'uid', 'cards_order'],
                        how='left')

        def get_cardsid_to_type(row):
            cardsid_list = str(row["cards"]).split(",")  # 出牌cards_id 的列表
            cards_id_type_list = []
            for cards_id in cardsid_list:
                try:
                    cards_ids = str(row["last_cards_id"]).split("||")
                    cards_id_type_index = [cards_id in inner_str.split("|") for inner_str in cards_ids]
                    cards_id_type_index = [cards_id in inner_str.split("|") for inner_str in cards_ids].index(1)
                    # cards_id_type_index = [cards_id in inner_list for str_inner in str(row["last_cards_id"]).split("||") for
                    #                            inner_list in str(str_inner).split("|")].index(1)
                    cards_id_type_list.append(
                        str(row["last_cards_type"]).split("||")[cards_id_type_index].split("|")[0])
                except ValueError:
                    cards_id_type_list.append('0')
            return "|".join(cards_id_type_list)

        # 筛选出牌的情况
        data_lead_cards = data.loc[(data.loc[:, "type"] > 0) & (data.loc[:, "cards_order"] != 1) & data.loc[:,
                                                                                                   'leftcards_nums'] != 0].reset_index(
            drop=True)
        # 取拆牌来源
        data_lead_cards.loc[:, "source"] = data_lead_cards.apply(lambda row: get_cardsid_to_type(row), axis=1)

        if data_lead_cards.shape[0]:
            write_data(data_lead_cards, filedir=result_dir, filename='chaipai_result', index=False)


def multiple_replace(text, adict):
    rx = re.compile('|'.join(map(re.escape, adict)))

    def one_xlat(match):
        return adict[match.group(0)]

    return rx.sub(one_xlat, text)


def group_chaipai_result(detail_result_dir):
    """聚合多个拆牌结果文件"""
    # result_dir = os.path.join(tmpdatadir1, 'detail_result')
    result_dir = detail_result_dir
    allfiles = [file for file in os.listdir(result_dir) if file.startswith("chaipai_result")]
    df = pd.DataFrame()
    for file in tqdm(allfiles):
        basic_cols = ["startguid", 'uid', 'playtime_unix', "type", 'label_uid', "num_show", "cards", 'rank',
                      "leftcards_nums", 'cards_order', 'cards_id', "cards_type", 'last_cards_id', "last_cards_type",
                      'source']
        tmp_df = pd.read_csv(os.path.join(result_dir, file), encoding="gbk", usecols=basic_cols)
        df = df.append(tmp_df, sort=False)

    cards_type_name_map = {
        '0': "未知",
        '1': "单张",
        '2': "对子",
        '4': "三张",
        "32": "三带二",
        '64': "顺子",
        '256': "连对",
        '512': "钢板",
        '4096': "炸弹4-5张",
        '16384': "同花顺",
        '32768': "超级大炸弹6-10张",
        '524288': "4王",
    }

    if df.shape[0]:
        df.loc[:, "type"] = df.loc[:, "type"].astype(str)
        df.loc[:, "type"] = df.loc[:, "type"].map(cards_type_name_map)
        df.loc[:, "source"] = df.loc[:, "source"].apply(
            lambda x: ",".join([x.replace(x, cards_type_name_map[x]) for x in str(x).split("|")]))
        write_data(df, filedir=result_dir, filename='all_chaipai_result')
        return df
    else:
        return None


if __name__ == '__main__':
    first_init()
    day_dir_names = [dir_name for dir_name in os.listdir(tmpdatadir1) if dir_name.startswith("20190720")]
    print(day_dir_names)
    for day_dir in day_dir_names:
        day_dir_path = os.path.join(tmpdatadir1, day_dir)
        print(day_dir_path)
        result_dir = os.path.join(tmpdatadir1, day_dir, 'detail_result')
        print(result_dir)
        # chaipai_apart(result_dir)  # 旧版拆牌来源
        chaipai_apart_new(result_dir)  # 新版拆牌来源
        chaipai_result = group_chaipai_result(result_dir)

        if chaipai_result is not None:
            # data = pd.read_csv(r"D:\projectsHome\ur.ctwl\tmpdata1\detail_result\chaipai_result_20190711111958.csv",
            #                    encoding='gbk')

            # 将结果继续拆分为 用户与机器人
            # chaipai_result = pd.read_csv("all_chaipai_result20190627_20190711154829.csv", encoding='gbk')
            # robot_id = pd.read_excel("机器人.xlsx")
            robot_id = pd.read_excel("robots_ids.xlsx")
            robot_ids = robot_id.loc[:, "用户ID"].unique()  # 机器人的uid

            robot_chaipai = chaipai_result.loc[chaipai_result.loc[:, 'uid'].isin(robot_ids)].reset_index(drop=True)
            robot_chaipai.to_csv(os.path.join(day_dir_path, f"robot_chaipai_result_{day_dir}.csv"), index=False,
                                 encoding='gbk')
            not_robot_chaipai = chaipai_result.loc[~chaipai_result.loc[:, 'uid'].isin(robot_ids)].reset_index(drop=True)
            not_robot_chaipai.to_csv(os.path.join(day_dir_path, f"not_robot_chaipai_result_{day_dir}.csv"), index=False,
                                     encoding='gbk')
