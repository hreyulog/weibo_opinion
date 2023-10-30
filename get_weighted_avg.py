import json
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import Counter


def get_weighted_avg(dim2list):
    sum_all = 0
    cont = 0
    for row in dim2list:
        if row[1] == 0 or row[1] == None:
            row[1] = 1
        sum_all += row[0] * row[1]
        cont += row[1]
    if sum_all / cont == 0:
        return 0.5
    return sum_all / cont


def getmonday(st):
    date = datetime.strptime(st, '%Y-%m-%d')
    week_start = date - timedelta(days=date.weekday())
    return week_start.strftime("%Y-%m-%d")


def get_oneweek_list(dic):
    dict_byday = {}
    for time in dic:
        monday = getmonday(time)
        if monday not in dict_byday:
            dict_byday[monday] = [dic[time]]
        else:
            dict_byday[monday].append(dic[time])
    return dict_byday


def main(user_id):
    dict_time = {}
    dict_avg = {}
    dict_like = {}
    dict_week_avg = {}
    with open(user_id + 'output.json', 'r', encoding='utf-8') as reader:
        for row in reader:
            json_row = json.loads(row)
            dict_time[json_row['time']] = json_row['comment_score']
            dict_like[json_row['time']] = int(json_row['likes'])
    for time in dict_time:
        dict_avg[time.split(' ')[0]] = (get_weighted_avg(dict_time[time]), dict_like[time])
    week_dict = get_oneweek_list(dict_avg)
    for week in week_dict:
        dict_week_avg[week] = get_weighted_avg(week_dict[week])
    return dict_week_avg


if __name__ == "__main__":

    dict_user_id = {}
    with open('list_bozhu.txt', 'r', encoding='utf-8') as reader:
        for row in reader:
            id = row.split()[1]
            dict_user_id[id] = main(id)
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
        print(i, len(y))
        # plt.plot(x, y)
        # plt.show ()
        ax.plot(x, y)
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
    res = []
    for i in dict_user_id:
        for j in dict_user_id[i]:
            res.append(j)
    res_dict = Counter(res)
    print(res_dict)

    plt.show()  # 图形可视化
