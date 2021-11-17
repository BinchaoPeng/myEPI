import calendar
import csv
import datetime
import json

import numpy as np
import requests


def get_day_list(startday, dayNum):
    startDay = datetime.datetime.strptime(startday, "%Y%m%d")
    print(startDay.strftime("query %Y-%m-%d"))
    day_list = [(startDay + datetime.timedelta(days=days)).strftime("%Y%m%d") for days in range(0, dayNum)]
    print(day_list)
    return day_list


def get_week_list(month):
    month_day_num = calendar.monthrange(2021, month)[1]
    first_weekday = calendar.monthrange(2021, month)[0]
    second_week_first_index = 7 - first_weekday
    week_start_list = [1]
    week_end_list = []
    for idx in range(second_week_first_index, month_day_num, 7):
        week_start_list.append(idx + 1)
        week_end_list.append(idx)
    print(week_start_list)
    week_end_list.append(month_day_num)
    print(week_end_list)
    return week_start_list, week_end_list


def get_dataList(startDate, endDate):
    payload = {"apiCode": "ab4afc14",
               "dataMap": {"startDate": startDate, "endDate": endDate, "offset": 0, "sort": "totalScore",
                           "pageSize": 40, "order": "desc", "isActivate": "", "deptId": 150015482586,
                           "orgGrayId": "num4MmTOnNagGN2Mlw7mTw=="}}
    response = requests.post(url='https://odrp.xuexi.cn/report/commonReport', json=payload, headers=headers)
    # 设置响应字符集
    # response.encoding = "utf-8"
    # 打印状态码、json字符串
    print(response.status_code)
    # print(response.json())
    # print(type(json.loads(response.json()["data_str"])))
    # print(json.loads(response.json()["data_str"]))
    data_dict = json.loads(response.json()["data_str"])
    data_list = data_dict["dataList"]["data"]
    """
    [{'rangeRealScore': 53, 'deptNames': '其他,研究生计算机技术专业第二党支部党员组', 
        'scoreMonth': 346000, 'rangeScore': 53, 'deptIds': '-1,150015482586', 'userName': '宋卓卿', 'userId': 604565936, 
        'totalScore': 41197, 'orgId': 150019702803, 'isActivate': 1}, 
        ...
     {'rangeRealScore': 15, 'deptNames': '其他,研究生计算机技术专业第二党支部党员组', 
     'scoreMonth': 71000, 'rangeScore': 15, 'deptIds': '-1,150015482586', 'userName': '潘波', 'userId': 151002277811, 
     'totalScore': 476, 'orgId': 150019702803, 'isActivate': 1}]
    """
    return data_list


def get_week_csv_data(startday, dayNum):
    day_list = get_day_list(startday=startday, dayNum=dayNum)

    data_all = {}
    for day in day_list:
        startDate = day
        endDate = day
        # time.sleep(0.1)
        flag = True
        data_list = None

        while (flag):
            try:
                data_list = get_dataList(startDate, endDate)
                flag = False
            except json.decoder.JSONDecodeError:
                print("connection failed,check your code,especially cookie!!!")
                exit()
            except Exception:
                print("request again")

        # data_list = get_dataList(startDate, endDate)
        # print(data_list)
        a_day_score = {}
        for person in data_list:
            userName = person["userName"]
            # 计入年度积分
            rangeRealScore = person["rangeRealScore"]
            # 计入总积分
            rangeScore = person["rangeScore"]
            a_day_score.update({userName: rangeRealScore})
        data_all.update({day: a_day_score})
    # print(data_all)

    header = ["姓名"]
    header.extend(list(data_all.keys()))
    header.extend(["总计", "活跃天数"])
    # print("header:", header)
    score_col_list = []
    userNames = list(data_all[list(data_all.keys())[0]].keys())
    score_col_list.append(userNames)
    for a_day_data in data_all.values():
        col = list(a_day_data.values())
        score_col_list.append(col)
    score_col_list = np.array(score_col_list).T
    # print(score_col_list)

    rows = []
    for p in score_col_list:
        row = []
        row.extend(p)
        score_sum = sum([int(s) for s in p[1:]])
        activate_day = len(p) - list(p).count("0") - 1
        row.append(score_sum)
        row.append(activate_day)
        rows.append(row)

    print(header)
    print(np.array(rows))
    return header, rows


def get_month_csv_data(month):
    week_start_list, week_end_list = get_week_list(month)
    header = []
    col = []
    userNames = []
    header.append("姓名")
    for start, end in zip(week_start_list, week_end_list):
        start_day, num_day = "2021" + str(month) + str(start), end - start + 1
        _, row = get_week_csv_data(start_day, num_day)
        row = np.array(row).T
        userNames = list(row[0])
        col.append(list(row[-2]))
        col.append(list(row[-1]))
        header.append(str(month) + "." + str(start) + "-" + str(month) + "." + str(end))
        header.append("活跃天数")
    col.insert(0, userNames)
    header.extend(["总计", "活跃天数"])
    # print("headers:", headers)
    # print(np.array(col).T)

    rows = []
    for p in np.array(col).T:
        row = []
        row.extend(p)
        score_sum = sum([int(s) for s in p[1::2]])
        activate_day = sum([int(s) for s in p[2::2]])
        row.append(score_sum)
        row.append(activate_day)
        rows.append(row)
    print("headers:", header)
    print(np.array(rows))
    return header, rows


def writeCSV(header, rows, file_name):
    with open(file_name, 'wt', newline='')as f:
        f_csv = csv.writer(f, delimiter=",")
        f_csv.writerow(header)
        f_csv.writerows(rows)
        f.close()
    print(file_name, "write over!!!")


Cookie = "__UID__=c4c55890-2c31-11ec-892b-adf7df2a7e3e; csrf_token=26291755306060181636354394142; aliyungf_tc=94c96a98b568cd5429cb3ad19e6784e4f6443648c0688023d3065ae084578c7d; tmzw=1636376526030; zwfigprt=2832750596d85e938a4f961e0583bb6f; acw_tc=707c9f6916364335139084610e56c6e27456c78cf6e340271b204e7b6b815b; token_=7d030a1e65-ac186299-88e64f-2d03de2e-910b048f-a352-4d40-b450-4c68a72fafde"

headers = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh-TW;q=0.9,zh;q=0.8",
    "Connection": "keep-alive",
    "Content-Length": "215",
    "Content-Type": "application/json;charset=UTF-8",
    "Cookie": Cookie,
    "Host": "odrp.xuexi.cn",
    "Origin": "https://study.xuexi.cn",
    "Pragma": "no-cache",
    "Referer": "https://study.xuexi.cn/",
    "sec-ch-ua": "'Google Chrome';v='95', 'Chromium';v='95', ';Not A Brand';v='99;",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "'Linux'",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36",
}


def get_week_reporter(start_day, num_day, file_name):
    header, rows = get_week_csv_data(start_day, num_day)
    writeCSV(header, rows, file_name)


def get_month_reporter(month, file_name):
    header, rows = get_month_csv_data(month)
    writeCSV(header, rows, file_name)


# if __name__ == '__main__':
#     get_month_reporter(month=10, file_name="./xuexi.csv")

import argparse

parser = argparse.ArgumentParser(description="XueXiQiangGuo script")
parser.add_argument('--t', type=str, default="W")
parser.add_argument('--sd', type=str, default=None)
parser.add_argument('--nd', type=int, default=7)
parser.add_argument('--f', type=str, default="./xuexi.csv")
parser.add_argument('--m', type=int, default=None)
args = parser.parse_args()

if args.t == "W":
    get_week_reporter(start_day=args.sd, num_day=args.nd, file_name=args.f)
elif args.t == "M":
    get_month_reporter(month=args.m, file_name=args.f)
else:
    print("param '--t'(type) must be 'W'(week) or 'M'(month)! ")
