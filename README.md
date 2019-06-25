# ur.ctwl
basic data feature engineering

基础特征处理

# 处理逻辑
* 手牌牌型拆解
    + cus == tb_customization , show == tb_showcards
    + show拿到每回合出牌，构建出累积出牌序列，将累积出牌序列填充未出牌时刻
    + cus拿到初始牌，填充初始牌列，初始牌 - 累积出牌序列的差集 = 剩余牌集 
    + 基于剩余牌集构建该牌集的唯一序列，调用【牌力值程序】，得到 牌型组合 + 牌力值
    + 基于牌型组合进行拆分，拿到各牌型数量 + 各牌型的最大牌值（用于与出牌牌值比较，判断是否需要出炸）
* 出牌牌值
    + 调用【牌力值程序】，拿到该出牌牌型的牌值

* 统计出炸情况
    +  基于出牌时间show.playtime_unix拿到每局前四手牌后构建出座位号 seat_order
    +  基于座位号 seat_order 和 cards_order （都为第一手）则为开局第一手（用于判断第一手出牌且是炸弹）
    +  确定如何统计及记录的统计值
        - 出牌方rival【剩余手牌数】
        - 出牌方出牌类型
        - 出牌方队友rival_pair 【剩余手牌数】
        - 队友 pair 【剩余手牌数】
        - 自己剩余牌型
        - 自己剩余炸弹数
    + 结果标记
        - 是否需要出炸 need_bomb == occurrence_times 
        - 是否出炸（直接基于出牌 num ） label_bomb == lead_bomb_times
    +  聚合统计值得到最终结果
        - 分批拿到部分对局的标记数据
        - 统计批次标记数据中的结果标记和
        - 合并各批次结果标记和
        - 计算概率值

# 数据映射
## 房间区分
- 17743  经典新手
- 17744  经典新手
- 8136  经典初级
- 10321  经典中级
- 10163  经典高级
- 18934  经典大师
- 19015  不洗牌中级
- 18149  不洗牌新手
- 18148  不洗牌初级
- 9533积分场

## 中英文名称对照字典
+ 列名称[https://github.com/firewang/ur.ctwl/blob/master/robot_bomb_colnames.txt](https://github.com/firewang/ur.ctwl/blob/master/robot_bomb_colnames.txt)
+ 统计值名称[https://github.com/firewang/ur.ctwl/blob/master/robot_bomb_statistic_situations.txt](https://github.com/firewang/ur.ctwl/blob/master/robot_bomb_statistic_situations.txt)