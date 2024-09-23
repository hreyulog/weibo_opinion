import json
from datetime import datetime
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号


def get_weighted_avg(dim2list):
    sum_all = 0
    cont = 0
    for row in dim2list:
        row = list(row)
        if row[1] is None:
            row[1] = 1
        else:
            row[1] += 1
        sum_all += row[0] * row[1]
        cont += row[1]
    if cont==0:
        return 0.5
    if sum_all / cont == 0:
        return 0.5
    return sum_all / cont


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


def commen_value(user_id):
    dict_time = {}
    dict_avg = []
    dict_like = {}
    dict_week_avg = {}

    with open(user_id + 'output_30.json', 'r', encoding='utf-8') as reader:
        for row in reader:
            json_row = json.loads(row)
            dict_time[json_row['time']] = json_row['comment_score']
            dict_like[json_row['time']] = int(json_row['likes'])
    for time in dict_time:
        dict_avg.append((time.split(' ')[0], (get_weighted_avg(dict_time[time]), dict_like[time])))
    week_dict = get_oneweek_list(dict_avg)
    for week in week_dict:
        dict_week_avg[week] = get_weighted_avg(week_dict[week])
    return dict_week_avg


def blog_value(user_id):
    dict_res = {}
    with open(user_id + 'output_70.json', 'r', encoding='utf-8') as reader:
        for row in reader:
            json_row = json.loads(row)
            dict_res[json_row['time'].split()[0]] = json_row['content_score']

    return dict_res


if __name__ == "__main__":
    dict_user_id = {}
    name_user = {}
    with open('list_bozhu.txt', 'r', encoding='utf-8') as reader:
        for row in reader:
            id = row.split()[1]
            name = row.split()[0]
            dict_user_id[id] = commen_value(id)
            name_user[id] = name
    fig, ax = plt.subplots()
    ind = 1
    for i in dict_user_id:
        # xs = [{datetime.strptime(d, '%Y-%m-%d').date():d} for d in dict_user_id[i]]
        xs = {}
        for d in dict_user_id[i]:
            xs[d] = datetime.strptime(d, '%Y-%m-%d').date()

        xs_sorted = sorted(xs.items(), key=lambda a: a[1])
        y = []
        x = []
        for zz in xs_sorted:
            x.append(zz[1])
            y.append(dict_user_id[i][zz[0]])
        # plt.plot(x, y)
        # plt.show ()
        # ax.plot(x, y)
        ind += 1
    time_list = []
    for i in dict_user_id:
        for j in dict_user_id[i]:
            time_da = datetime.strptime(j, '%Y-%m-%d').date()
            time_list.append(j)

    time_s_set = sorted(list(set(time_list)))
    time_s_dict = {}
    for i in dict_user_id:
        for j in dict_user_id[i]:
            if j not in time_s_dict:
                time_s_dict[j] = []
            else:
                time_s_dict[j].append((i, dict_user_id[i][j]))
    res = {}
    for i in dict_user_id:
        x = []
        for j in dict_user_id[i]:
            if j not in res:
                res[j] = {i: dict_user_id[i][j]}
            else:
                res[j][i] = dict_user_id[i][j]
    time_dict = {}
    for time in res:
        if len(res[time]) >= 7  and time != '2023-05-01':
            time_dict[time] = res[time]
    print(time_dict)
    sorted_dict = sorted(dict_user_id.keys())
    for user_id in sorted_dict:
        x = []
        y = []
        for time in time_dict:
            time_dt = datetime.strptime(time, '%Y-%m-%d').date()
            # plt.axvline(time_dt)
            x.append(time_dt)
            y.append(time_dict[time][user_id])
        plt.plot(x, y, label=name_user[user_id])
    plt.legend(loc=2)
    plt.show()  # 图形可视化
