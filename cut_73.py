import json
from datetime import datetime
import matplotlib.pyplot as plt
import random
from time import sleep
def getmonday(st):
    date = datetime.strptime(st, '%Y-%m-%d')
    month = str(date.month)
    if len(month) == 1:
        month = '0' + month
    if (date.day - 1) // 15 == 0:
        return f'{date.year}-{month}-01'
    else:
        return f'{date.year}-{month}-16'
    # print((date.day-1)//15)
    # week_start = date - timedelta(days=date.weekday())
    # print(week_start.strftime("%Y-%m-%d"))
    # return week_start.strftime("%Y-%m-%d")


def get_oneweek_list(dic):
    dict_byday = {}
    for time in dic:
        monday = getmonday(time[0])
        if monday not in dict_byday:
            dict_byday[monday] = [time[1]]
        else:
            dict_byday[monday].append(time[1])
    return dict_byday


def split_list(input_list, split_ratio=0.7):
    # 打乱列表顺序
    random.shuffle(input_list)

    # 计算70%的部分的大小
    split_point = int(len(input_list) * split_ratio)

    # 切分列表
    list_70 = input_list[:split_point]
    list_30 = input_list[split_point:]

    return list_70, list_30

def commen_value(user_id):
    dict_time = {}
    dict_like = {}
    dict_week_avg = {}
    with open(user_id+'output_70.json','w',encoding='utf-8') as writer70:
        with open(user_id+'output_30.json','w',encoding='utf-8') as writer30:
            with open(user_id + 'output.json', 'r', encoding='utf-8') as reader:
                for row in reader:
                    json_row = json.loads(row)
                    comment_score = json_row['comment_score']
                    list_70, list_30 = split_list(comment_score)
                    print(len(comment_score))
                    print(len(list_70))
                    print(len(list_30))
                    json_row['comment_score']=list_70
                    writer70.write(json.dumps(json_row,ensure_ascii=False))
                    writer70.write('\n')
                    json_row['comment_score']=list_30
                    writer30.write(json.dumps(json_row,ensure_ascii=False))
                    writer30.write('\n')
            print(dict_time)

            return dict_week_avg


if __name__ == "__main__":
    dict_user_id = {}
    name_user = {}
    with open('list_bozhu.txt', 'r', encoding='utf-8') as reader:
        for row in reader:
            id = row.split()[1]
            name = row.split()[0]
            dict_user_id[id] = commen_value(id)
            name_user[id] = name
