# -*- encoding: utf-8 -*-
# @Version : 1.0  
# @Time    : 2019/8/12 10:42
# @Author  : wanghd
# @note    : 按牌力值方案弃牌（3，5，7，9张），对比前后牌力值变化情况 ; 基于robot_result出炸统计结果

import os
import subprocess
import pandas as pd
import time
import uuid


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


def write_data(df, filedir, filename='qipai', index=False):
    if not os.path.exists(filedir):
        os.mkdir(filedir)
    current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
    filename = f'{filename}_{current_time}.csv'
    df.to_csv(os.path.join(filedir, filename), index=index, header=True, encoding='gbk')


def calculate_cards_value(df, cards_col, rank_col):
    """调用牌力值程序得到牌力值"""
    base_path = os.path.abspath("F:/CardsValue/CardsValue")
    input_file = os.path.join(base_path, 'Input.txt')
    output_file = os.path.join(base_path, 'Output.txt')
    if os.path.exists(input_file):
        os.remove(input_file)
    if os.path.exists(output_file):
        os.remove(output_file)

    # 写入牌面信息 到 input
    with open(input_file, 'a') as f:
        for rowid in range(df.shape[0]):
            f.write(df.at[rowid, "uID"])
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

    # 执行 CalculatorCardsValue.exe 获得拆牌信息
    subprocess.call(os.path.join(base_path, 'CalculatorCardsValue2dir.exe'))

    # 读取拆牌信息
    df_cardsvalue_cols = ["uID", 'cards_value', 'cards_id', 'cards_type']  # 唯一标识，牌力值，ID组，类型组
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
                    df_cardsvalue.at[index - 1, "uID"] = line.strip()
                if index % 5 == 3:
                    df_cardsvalue.at[index - 3, 'cards_value'] = line.strip()
                if index % 5 == 4:
                    df_cardsvalue.at[index - 4, 'cards_id'] = line.strip()
                if index % 5 == 0:
                    df_cardsvalue.at[index - 5, 'cards_type'] = line.strip()
                # print(index, line)

    df_cardsvalue.drop_duplicates(inplace=True)

    df = pd.merge(df, df_cardsvalue, on=["uID"], how='left', copy=False)  # 合并牌力值拆牌信息
    return df.loc[:, ["uID", 'cards_value', "cards_id"]]  # 返回牌力值和牌组ID


def perform_calculated_cards_value(df, cards_col, rank_col, new_names_list):
    """执行牌力值程序计算牌力值, 拿到不同弃牌数的牌力值"""
    # 级牌和起始手牌转化为 牌力值
    df_first_not_null = calculate_cards_value(df, cards_col, rank_col)
    df_first_not_null.rename(columns={"cards_value": new_names_list[0],
                                      "cards_id": new_names_list[1]}, inplace=True)
    df = pd.merge(df, df_first_not_null, how='left', on='uID')  # 合并弃牌后剩余牌的牌力值及牌组ID
    return df


if __name__ == '__main__':
    base_dir = os.path.abspath(r"F:\raw_data_dir\20190703\detail_result")
    # 初始df, 存储牌力值结果
    df = pd.DataFrame()
    for file in os.listdir(base_dir):
        if file.startswith("robot_result"):
            robot_result = get_data(file, base_dir)
            robot_result = robot_result.loc[:,
                           ["startguid", 'uid', 'rank', 'leftcards_nums', 'cards_value', 'cards_id']].copy()
            robot_result.drop_duplicates(inplace=True)
            robot_result = robot_result.reset_index(drop=True)
            robot_result = robot_result.query("leftcards_nums == 27").reset_index(drop=True)
            if robot_result.shape[0]:
                df = pd.concat([robot_result, df])
        if df.shape[0] > 1000:
            df.drop_duplicates(inplace=True)
            df = df.reset_index(drop=True)
            break
    df.rename(columns={'cards_value':"init_cards_value", 'cards_id':"init_cards_id"}, inplace=True)
    df.loc[:, 'qi3pai_cards_id'] = df.loc[:, 'init_cards_id'].apply(lambda x: x.replace("||", "|").split("|")[3:])
    df.loc[:, 'qi5pai_cards_id'] = df.loc[:, 'init_cards_id'].apply(lambda x: x.replace("||", "|").split("|")[5:])
    df.loc[:, 'qi7pai_cards_id'] = df.loc[:, 'init_cards_id'].apply(lambda x: x.replace("||", "|").split("|")[7:])
    df.loc[:, 'qi9pai_cards_id'] = df.loc[:, 'init_cards_id'].apply(lambda x: x.replace("||", "|").split("|")[9:])
    df.loc[:, 'init_cards_id'] = df.loc[:, 'init_cards_id'].apply(lambda x: x.replace("||", "|").split("|"))
    df.loc[:, 'uID'] = pd.Series([str(uuid.uuid1()) for _ in range(df.shape[0])])  # 生成唯一标识

    qipai_cols = ['qi3pai_cards_id','qi5pai_cards_id','qi7pai_cards_id','qi9pai_cards_id']
    # qipai_cols = ['qi3pai_cards_id']
    for qipai in qipai_cols:
        new_name = [qipai.split('_')[0] + 'cards_value' + 'new', qipai.split('_')[0] + 'cards_id' + 'new']
        df = perform_calculated_cards_value(df, qipai, 'rank', new_name)
    write_data(df, filedir=base_dir)  # 写出文件
