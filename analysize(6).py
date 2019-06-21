# -*- coding: utf-8 -*-

import pandas as pd

#预处理局数信息，把ResultGuid相同的放入以ResultGuid为key值的字典里
def preResultGuid( boutdict, csvdata):
    line = []
    for index in csvdata.index:
        print(str(index))
        line = csvdata.loc[index].values
        #print(line[0])                                          
        if(line[0] in boutdict):                  #如果有值，在列表后插入值
            boutdict[line[0]].append(line)
        else:
            boutdict[line[0]] = []                 #如果没有值，新建并插入值
            boutdict[line[0]].append(line)  

def sortdctlist(listinfo, foo): 
    #    print(str(index)  + ":" + str(listinfo[index][11]))                                     #对存入值为字典的列表进行排序
    listinfo.sort(key = foo)

def writetocsv(cardsdict, writemode, needheader):
    DF = pd.DataFrame(cardsdict, columns=["resultguid", "用户id", "剩余手牌数", "单牌数", "炸弹数", "同花顺数", "上个出牌玩家", "上个出牌牌型", "上个出牌大小", "上个出牌num", "出牌类型", "出牌牌型", "牌大小", "出牌num", "对家剩余手牌数", "上家剩余手牌数", "下家剩余手牌数"])
    DF.to_csv("out.csv", mode=writemode, header=needheader, index = False)

def calccardindex(cardid):
    card54id = cardid % 54
    layindex = 0
    if(card54id < 52):
        layindex = card54id % 13
    elif (card54id == 52):
        layindex = 13
    else:
        layindex = 14

    return layindex
    
def predealcards(index, cardsvalue, customdict):
    layout = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    userid = cardsvalue[index][10]
    for i in range(len(customdict[userid])):
        cardid = int(customdict[userid][i])
        layindex = calccardindex(cardid)
        layout[layindex] = layout[layindex] + 1

    return layout

def predealflushcards(index, cardsvalue, customdict):
    laydiamondout = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    layclubout = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    layheartout = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    layspadeout = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    userid = cardsvalue[index][10]
    for i in range(len(customdict[userid])):
        cardid = int(customdict[userid][i])
        if(cardid%54 < 13):
            layindex = calccardindex(cardid)
            laydiamondout[layindex] = laydiamondout[layindex] + 1
        elif(cardid%54 < 26):
            layindex = calccardindex(cardid)
            layclubout[layindex] = layclubout[layindex] + 1
        elif(cardid%54 < 39):
            layindex = calccardindex(cardid)
            layheartout[layindex] = layheartout[layindex] + 1
        elif(cardid%54 < 52):
            layindex = calccardindex(cardid)
            layspadeout[layindex] = layspadeout[layindex] + 1
        else:
            pass

    return laydiamondout, layclubout, layheartout, layspadeout
    

def calcsignlecard(cardsdict, index, cardsvalue, customdict):
    layout = predealcards(index, cardsvalue, customdict)
    signlenum = 0
    for i in range(0, 15):
        if layout[i] == 1:
            signlenum = signlenum + 1

    templist = []
    templist.append(signlenum)
    cardsdict["单牌数"] = templist

def calcbombcard(cardsdict, index, cardsvalue, customdict):
    layout = predealcards(index, cardsvalue, customdict)
    bombnum = 0
    for i in range(0, 15):
        if layout[i] >= 4:
            bombnum = bombnum + 1

    if(layout[13] == 2 and layout[14] == 2):
        bombnum = bombnum + 1
    
    templist = []
    templist.append(bombnum)
    cardsdict["炸弹数"] = bombnum

def calcflushcard(cardsdict, index, cardsvalue, customdict):
    laydiamondout, layclubout, layheartout, layspadeout = predealflushcards(index, cardsvalue, customdict)
    flushnum = 0
    for i in range(0, 9):
        count = 1
        for j in range(i + 1, 13):
            if (laydiamondout[j] >= 1):
                count = count + 1
                if (count >= 5):
                    flushnum = flushnum + 1
                    break
            else:
                break
        
    for i in range(0, 9):
        count = 1
        for j in range(i + 1, 13):
            if (layclubout[j] >= 1):
                count = count + 1
                if (count >= 5):
                    flushnum = flushnum + 1
                    break
            else:
                break

    for i in range(0, 9):
        count = 1
        for j in range(i + 1, 13):
            if (layheartout[j] >= 1):
                count = count + 1
                if (count >= 5):
                    flushnum = flushnum + 1
                    break
            else:
                break

    for i in range(0, 9):
        count = 1
        for j in range(i + 1, 13):
            if (layspadeout[j] >= 1):
                count = count + 1
                if (count >= 5):
                    flushnum = flushnum + 1
                    break
            else:
                break    

    templist = []
    templist.append(flushnum)
    cardsdict["同花顺数"] = flushnum


def calchandcardsnum(cardsdict, index, cardsvalue, customdict, duijiainfo, shangjiainfo, xiajiainfo):
    userid = cardsvalue[index][10]
    templist = []
    #if(userid in customdict):
    templist.append(len(customdict[userid]))
    cardsdict["剩余手牌数"] = templist

    templist = []
    print(duijiainfo[userid])
    #print(len(customdict[duijiainfo[userid]]))
    #print(customdict[duijiainfo[userid]])
    if(duijiainfo[userid] in customdict):
        templist.append(len(customdict[duijiainfo[userid]]))
    cardsdict["对家剩余手牌数"] = templist

    templist = []
    #print(shangjiainfo[userid])
    if(shangjiainfo[userid] in customdict):
        templist.append(len(customdict[shangjiainfo[userid]]))
    cardsdict["上家剩余手牌数"] = templist

    templist = []
    #print(xiajiainfo[userid])
    if(xiajiainfo[userid] in customdict):
        templist.append(len(customdict[xiajiainfo[userid]]))
    cardsdict["下家剩余手牌数"] = templist

    #print(cardsvalue[index][18])
    templist = str(cardsvalue[index][18]).split(",")
    customdict[userid] = list(set(customdict[userid]) - set(templist))
    #print(customdict[userid])

def calcselfinfo(cardsdict, index, cardsvalue):
    bc = 1
    templist = []
    #print(cardsvalue[index][15])
    if(cardsvalue[index][15] == 1):
        templist = []
        templist.append("不出")
        cardsdict["出牌类型"] = templist
    else:
        if(index == 0):
            templist = []
            templist.append("出牌")
            cardsdict["出牌类型"] = templist
        else:
            while(True):
                #print(cardsvalue[index - bc][18])
                if(str(cardsvalue[index - bc][18]) != "nan"):               #len(cardsvalue[index - bc][18]) > 0):                                          #找到上一个有出牌信息的玩家
                    break
                else:
                    bc = bc + 1
            if (bc == 4):
                templist = []
                templist.append("出牌")
                cardsdict["出牌类型"] = templist
            else:
                templist = []
                templist.append("压牌")
                cardsdict["出牌类型"] = templist

        #cardsdict["出牌牌型"] = cardsvalue[index]["type_name"].decode('gbk').encode('utf-8')
        templist = []
        templist.append(cardsvalue[index][22])
        cardsdict["出牌牌型"] = templist
        templist = []
        templist.append("待定")
        cardsdict["牌大小"] = templist
        templist = []
        templist.append(cardsvalue[index][21])
        cardsdict["出牌num"] = templist

def calclastplayer(cardsdict, index, cardsvalue):
    #userid = cardsvalue[index][10]
    bc = 1
    templist = []
    if(index == 0):
        cardsdict["上个出牌玩家"] = [""]
        cardsdict["上个出牌牌型"] = [""]
        cardsdict["上个出牌大小"] = [""]
        cardsdict["上个出牌num"] = [""]
    else:
        while(True):
            if(str(cardsvalue[index - bc][18]) != "nan"):                                          #找到上一个有出牌信息的玩家
                if(bc == 1):
                    templist = []
                    templist.append("上家")
                    cardsdict["上个出牌玩家"] = templist
                elif(bc == 2):
                    templist = []
                    templist.append("对家")
                    cardsdict["上个出牌玩家"] = templist
                elif(bc == 3):
                    templist = []
                    templist.append("下家")
                    cardsdict["上个出牌玩家"] = templist
                else:
                    templist = []
                    templist.append("本家")
                    cardsdict["上个出牌玩家"] = templist
                #cardsdict["上个出牌牌型"] = cardsvalue[index - bc]["type_name"].decode('gbk').encode('utf-8')
                templist = []
                templist.append(cardsvalue[index - bc][22])
                cardsdict["上个出牌牌型"] = templist
                templist = []
                templist.append("待定")
                cardsdict["上个出牌大小"] = templist
                templist = []
                templist.append(cardsvalue[index - bc][21])
                cardsdict["上个出牌num"] = templist
                break
            else:
                bc = bc + 1

def buildnewinfo(cardsvalue, customvalue, writedict):
    tempcustomdict = {}
    duijiainfo = {}                         #对家id
    shangjiainfo = {}                       #上家id
    xiajiainfo = {}                         #下家id

    for index in range(len(customvalue)):                     #计算对家信息   
        selfuserid = customvalue[index][11]
        duijiauserid = selfuserid
        shangjiauserid = selfuserid
        xiajiauserid = selfuserid
        duijiauserid = customvalue[(index + 2) % 4][11]
        shangjiauserid = customvalue[(index + 3) % 4][11]
        xiajiauserid = customvalue[(index + 1) % 4][11]
        
        duijiainfo[selfuserid] = duijiauserid
        shangjiainfo[selfuserid] = shangjiauserid
        xiajiainfo[selfuserid] = xiajiauserid
    for index in range(len(customvalue)):
        tempcustomdict[customvalue[index][11]] = customvalue[index][24].split(",")
    print(len(cardsvalue))
    for index in range(len(cardsvalue)):
        tempcardsdict = {}
        templist = []
        templist.append(cardsvalue[index][0])
        tempcardsdict["resultguid"] = templist
        templist = []
        templist.append(cardsvalue[index][10])
        tempcardsdict["用户id"] = templist
        calcsignlecard(tempcardsdict, index, cardsvalue, tempcustomdict)           #计算单牌数
        calcbombcard(tempcardsdict, index, cardsvalue, tempcustomdict)             #计算炸弹数
        calcflushcard(tempcardsdict, index, cardsvalue, tempcustomdict)            #计算同花顺数
        calchandcardsnum(tempcardsdict, index, cardsvalue, tempcustomdict, duijiainfo, shangjiainfo, xiajiainfo)           #计算当前手牌数
        calcselfinfo(tempcardsdict, index, cardsvalue)               #计算自己的出牌信息
        calclastplayer(tempcardsdict, index, cardsvalue)             #计算上个玩家信息
        writetocsv(tempcardsdict, writedict["writemode"], writedict["needheader"]) 
        writedict["writemode"] = "a"
        writedict["needheader"] = False              

def analysize(boutcardsdict, boutcustomdict, writedict):
    for key in boutcardsdict.keys():
        foo = lambda s:int(s[11])
        cardsvalue = boutcardsdict[key]
        customvalue = {}
        print(key)
        if key in boutcustomdict:
            customvalue = boutcustomdict[key]
            if (len(customvalue) >= 4):
                sortdctlist(cardsvalue, foo)                                         #对每一局的信息按playetime_unix排序
                buildnewinfo(cardsvalue, customvalue, writedict)                       #对每局的信息进行处理，生成新的信息 
            else:
                print("boutcustomdict key len :" + str(key) + " not enough") 
        else:
            print("boutcustomdict key:" + str(key) + " lose")

def main():
    print("hello")
    cards_csv_data = pd.read_csv("cards_last.csv", low_memory=False, encoding = "utf-8")
    #cards_csv_data = pd.read_csv("cards-test.csv", encoding = "gbk")
    print("read cards_csv_data end")
    custom_csv_data = pd.read_csv("custom_last.csv", low_memory=False, encoding = "utf-8")
    #custom_csv_data = pd.read_csv("custom-test.csv", encoding = "gbk")
    print("read custom_csv_data end")

    boutcardsdict = {}
    boutcustomdict = {}

    writedict = {"writemode":"w", "needheader":"True"}

    preResultGuid(boutcardsdict, cards_csv_data)
    preResultGuid(boutcustomdict, custom_csv_data)
    analysize(boutcardsdict, boutcustomdict, writedict)

if __name__ == '__main__':
    main()
    print("end")
