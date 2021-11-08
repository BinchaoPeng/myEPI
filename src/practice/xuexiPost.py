import requests
import json
import datetime


def get_day_list(startday, dayNum):
    startDay = datetime.datetime.strptime(startday, "%Y%m%d")
    print(startDay.strftime("%Y%m%d"))
    day_list = [(startDay + datetime.timedelta(days=days)).strftime("%Y%m%d") for days in range(0, dayNum)]
    print(day_list)
    return day_list


def get_dataList(startDate, endDate):
    json_str = '''{"apiCode":"ab4afc14","dataMap":{
                    "startDate":"%s","endDate":"%s",
                    "offset":0,"sort":"totalScore",
                    "pageSize":40,
                    "order":"desc","isActivate":"","orgGrayId":"num4MmTOnNagGN2Mlw7mTw=="}}''' % (startDate, endDate)

    payload = json.loads(json_str)

    response = requests.post(url='https://odrp.xuexi.cn/report/commonReport', json=payload, headers=headers)
    # 设置响应字符集
    response.encoding = "utf-8"
    # 打印状态码、json字符串
    print(response.status_code)
    print(response.json())
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


Cookie = "__UID__=c4c55890-2c31-11ec-892b-adf7df2a7e3e; csrf_token=26291755306060181636354394142; aliyungf_tc=94c96a98b568cd5429cb3ad19e6784e4f6443648c0688023d3065ae084578c7d; tmzw=1636376526030; zwfigprt=2832750596d85e938a4f961e0583bb6f; token=063e8b63d83f45d89788c26a0565b9ee; acw_tc=2f6fc10216363801539292000e5568675deba48bef4362275747be130b5d9e; token_=7cffdd8b3d-ac1ef6c4-52b6a9-4e5ee240-5d841020-de7d-49de-8e15-46041b1f0980"
headers = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh-TW;q=0.9,zh;q=0.8",
    "Connection": "keep-alive",
    "Content-Length": "193",
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

day_list = get_day_list(startday="20211130", dayNum=1)

data_all = []
for day in day_list:
    startDate = day
    endDate = day
    data_list = get_dataList(startDate, endDate)
    print(data_list)
    a_day_score = []
    for person in data_list:
        userName = person["userName"]
        # 计入年度积分
        rangeRealScore = person["rangeRealScore"]
        # 计入总积分
        rangeScore = person["rangeScore"]
        a_day_score.append({userName: rangeRealScore})
    data_all.append(a_day_score)
print(data_all)
