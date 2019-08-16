# -*- encoding: utf-8 -*-
# @Version : 1.0  
# @Time    : 2019/8/16 11:17
# @Author  : wanghd
# @note    : 清洗robot_bomb v3 版本的错误统计数据 ; 修改main 函数中base_dir
import configparser
import os
from collections import Counter
from itertools import compress

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def write_data(df, filedir=os.getcwd(), filename='robot_bomb', index=False):
    if not os.path.exists(filedir):
        os.mkdir(filedir)
    current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
    filename = f'{filename}_{current_time}.csv'
    df.to_csv(os.path.join(filedir, filename), index=index, header=True, encoding='gbk')


def get_config_dict(filepath, filename, section_name):
    """读取配置文件中的字典"""
    config = configparser.ConfigParser()
    config.read(os.path.join(filepath, filename))
    config_dict = dict(config.items(section_name))
    # print(config_dict)
    return config_dict


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


def rival_leadcards_treatment2(df):
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
        leftcards_exclude = set(str(source_df.at[source_idx + plus_idx, 'leftcards_exclude']).split(","))
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
        cards_order_df = pd.DataFrame(list(range(1, gamedf.cards_order.max() + 1)), columns=["cards_order"])
        cards_order_df.loc[:, 'startguid'] = start_guid
        cards_order_seat_order = gamedf.loc[:, ['startguid', 'uid', 'seat_order', "playtime_unix"]].sort_values(
            by=["playtime_unix"], ascending=True)
        cards_order_seat_order = cards_order_seat_order.groupby(["startguid"]).head(4)
        cards_order_seat_order.drop(columns=["playtime_unix"], inplace=True)
        cards_order_df = pd.merge(cards_order_df, cards_order_seat_order, on=['startguid'], how='left')

        # cards_order_df.to_excel(f"F:/aaa/cards_order_df.xlsx", index=False)

        # print(cards_order_df.head())
        gamedf.drop(columns=["seat_order"], inplace=True)  # 删除原始的seat_order
        gamedf = pd.merge(gamedf, cards_order_df, on=["startguid", 'cards_order', 'uid'], how='right')

        # gamedf.to_excel(f"F:/aaa/gamedf.xlsx", index=False)

        gamedf = gamedf.sort_values(by=['uid', 'cards_order'], ascending=[True, True])
        gamedf.reset_index(drop=True, inplace=True)
        # 填充由于填补 cards_order 导致的缺失值
        for col in ["cards", 'num_show', 'leftcards_exclude', 'cards_id', 'cards_type', 'leftcards_exclude']:
            gamedf.loc[:, col] = gamedf.loc[:, col].fillna('0')
        for col in ['starttime_unix', "playtime_unix", 'rank', 'cards_init', 'num_init', 'label_uid', 'leftcards',
                    'leftcards_face']:
            gamedf.loc[:, col] = gamedf.loc[:, col].fillna(method='ffill')
        # 剩余的其他数值列: type【主要】, 剩余牌数，各牌型数量，其他选手牌数等
        # gamedf.loc[:, 'type'] = gamedf.loc[:, "type"].fillna(0)
        for col in gamedf.columns:
            gamedf.loc[:, col] = gamedf.loc[:, col].fillna(0)

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
        for idx in range(2, idx_length - 3):
            if gamedf.at[idx, 'type'] > 0:
                gamedf = get_rival_info(gamedf, info_idx=idx, plus_idx=1, game_start=False)
                if gamedf.at[idx + 1, 'type'] == 0 and gamedf.at[idx + 2, 'type'] == 0:
                    # 第二位(下家)， 不出，不出 的情况
                    gamedf = get_rival_info(gamedf, info_idx=idx, plus_idx=3, game_start=False)
        for idx in [idx_length - 3, idx_length - 2]:
            # 局倒数第三手，第二手的情况
            if gamedf.at[idx, 'type'] > 0:
                gamedf = get_rival_info(gamedf, info_idx=idx, plus_idx=1, game_start=False)

        # 出牌对手出牌的统计标记
        for idx in range(idx_length - 3):
            leadcard_type = gamedf.at[idx, 'type']  # 出牌类型
            if leadcard_type in [4096, 16384, 32768, 524288]:
                # 对手出炸弹
                gamedf = compare_bomb(gamedf, idx, 1)
                if gamedf.at[idx + 1, 'type'] == 0 and gamedf.at[idx + 2, 'type'] == 0:
                    # 第二位，不出，不出 的情况
                    gamedf = compare_bomb(gamedf, idx, 3)

            elif leadcard_type in [1, 2, 4]:
                # 对手出单张, 对子，三张
                gamedf = compare_cards(gamedf, type_card=leadcard_type, source_idx=idx, plus_idx=1, stage=True)
                if gamedf.at[idx + 1, 'type'] == 0 and gamedf.at[idx + 2, 'type'] == 0:
                    # 第二位队友(下家)的情况（不出，不出）
                    gamedf = compare_cards(gamedf, type_card=leadcard_type, source_idx=idx, plus_idx=3, stage=True)
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

        for idx in range(idx_length - 3, idx_length - 1):
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

        gamedf_cols = ['startguid', 'uid', 'cards_order', 'playtime_unix', "leftcards_nums_pair",
                       'rival_leftcards_nums', 'rival_leftcards_nums_pair',
                       "rival_cards_value", "rival_position", 'rival_leadcards_type', 'rival_leadcards_cards',
                       'rival_leadcards_num_show', 'need_bomb', 'label_bomb', ]

        gamedf = gamedf.loc[:, gamedf_cols]
        # gamedf.drop(columns=['seat_order', 'leftcards_nums', 'startguid', 'uid', ], inplace=True)
        statistic_df = statistic_df.append(gamedf, sort=False, ignore_index=True)  # concat会匹配index,ignore_index
        # statistic_df.to_excel(f"F:/aaa/gamedf_statistic.xlsx", index=False)

    # df = pd.concat([df, statistic_df], axis=1) # 按顺序匹配总是可能存在index的问题
    # dd = pd.merge(df, statistic_df, on='playtime_unix', copy=False)  # 根据出牌时间来匹配队友剩余牌数
    # dd.to_excel(f"F:/aaa/gamedf_merge_playtime.xlsx", index=False)
    df = pd.merge(df, statistic_df, on=['startguid', 'uid', 'cards_order'], copy=False)  # 根据匹配队友剩余牌数
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
        # write_data(statistic_df, filedir=os.path.join(tmpdatadir1, "20190709"), filename='statistic_result', )
    return statistic_df, data_length


if __name__ == '__main__':
    base_dir = os.path.abspath(r'D:\projectsHome\ur.ctwl\tmpdata1')
    day_dir_names = [dir_name for dir_name in os.listdir(base_dir) if dir_name.startswith("201907")]
    print(day_dir_names)
    for day_dir in tqdm(day_dir_names):
        day_dir_path = os.path.join(base_dir, day_dir)
        latest_file_name = [file for file in os.listdir(day_dir_path) if file.startswith("latest")][0]
        os.remove(os.path.join(day_dir_path, latest_file_name))
        print(day_dir_path)
        result_dir = os.path.join(base_dir, day_dir, 'detail_result')
        # result_dir = os.path.join(base_dir, day_dir, 'detail_result','detail_result')
        print(result_dir)
        statistic_files = [file for file in os.listdir(result_dir) if file.startswith("statistic")]
        for statistic_file in tqdm(statistic_files):
            os.remove(os.path.join(result_dir, statistic_file))

        for file in tqdm([file for file in os.listdir(result_dir) if file.startswith("robot_result")]):
            merge_data = pd.read_csv(os.path.join(result_dir, file))
            duoyu = ["leftcards_nums_pair", "rival_leftcards_nums", "rival_leftcards_nums_pair", "rival_cards_value",
                     "rival_position",
                     "rival_leadcards_type", "rival_leadcards_cards", "rival_leadcards_num_show", "need_bomb",
                     "label_bomb"]
            merge_data.drop(columns=duoyu, inplace=True)

            marker = rival_leadcards_treatment2(merge_data)
            # 重写robot_bomb文件
            # robot_result_filepath = os.path.join(result_dir, file)
            # os.remove(robot_result_filepath)
            # marker.to_csv(robot_result_filepath, encoding='gbk', index=False)
            chunk_df, _ = statistic_procedure_v2(marker)
            if chunk_df.shape[0]:
                robot_result_time = file.split('_')[-1]
                chunk_df.to_csv(os.path.join(result_dir, f"statistic_result_{robot_result_time}"), index=False,
                                encoding='gbk')

        files = [file for file in os.listdir(result_dir) if file.startswith("statistic")]
        df = pd.DataFrame()
        for statistic_file in tqdm(files):
            previous_data = pd.read_csv(os.path.join(result_dir, statistic_file), encoding='gbk')
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

        df.to_csv(os.path.join(day_dir_path, latest_file_name), encoding='gbk', index=False)
