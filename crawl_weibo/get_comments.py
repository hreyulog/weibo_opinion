import csv
import json
import re

import requests

user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36'
headers = {'User_Agent': user_agent,
           'cookie': "SCF=ArwBbgYGS-iqpxaFehhbJNxigFFCyq_Ea-EVTsARvUjaKRWcE-9NLsoOA_uw2d6RIXS9Jy5xV7lZl4q2uNxM1zg.; SSOLoginState=1697808790; ALF=1700400790; loginScene=102003; geetest_token=98293314bbe5407752e972b667dcf7f0; SUB=_2A25INvIYDeRhGeFG7FUU8yfOyDmIHXVr2J5QrDV6PUJbkdANLW3BkW1NeMEjPiH847_Xelv2J8X79yG8t8HT4TAu; MLOGIN=1; _T_WM=89205870901; M_WEIBOCN_PARAMS=luicode%3D20000174%26lfid%3D102803"
           }


def get_comments(user_id):
    dict_weibo = {}
    filename = user_id + '.csv'
    with open('weibo/' + filename, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            id_weibo, content, time, likes = row['\ufeff微博id'], row['微博正文'], row['发布时间'], row['点赞数']
            if 'huawei' in content or '华为' in content:
                dict_weibo[id_weibo] = {'content': content, 'time': time, 'likes': likes}
    with open('weibo/comment' + filename + '.json', 'w', encoding='utf-8') as writer:
        for id in dict_weibo:
            print(id)
            page = 1
            max_id = ''
            while True:
                if page == 1:  # 第一页，没有max_id参数
                    url = 'https://m.weibo.cn/comments/hotflow?id={}&mid={}&max_id_type=0'.format(id, id)
                else:  # 非第一页，需要max_id参数
                    if max_id == "0":  # 如果发现max_id为0，说明没有下一页了，break结束循环
                        print('max_id is 0, break now')
                        break
                    url = 'https://m.weibo.cn/comments/hotflow?id={}&mid={}&max_id_type=0&max_id={}'.format(id, id,
                                                                                                            max_id)
                r = requests.get(url, headers=headers)
                print(r.json())
                if r.json()['ok'] == 0:
                    break
                datas = r.json()['data']['data']
                for data in datas:
                    dr = re.compile(r'<[^>]+>', re.S)  # 用正则表达式清洗评论数据
                    text2 = dr.sub('', data['text'])
                    this_dict = {
                        'id': id,
                        'content': dict_weibo[id]['content'],
                        'time': dict_weibo[id]['time'],
                        'likes': dict_weibo[id]['likes'],
                        'comment': {'id': data['user']['id'], 'comment': text2, 'time': data['created_at'],
                                    'likes': data['like_count']},
                    }
                    writer.write(json.dumps(this_dict, ensure_ascii=False))
                    writer.write('\n')
                page += 1
                max_id = str(r.json()['data']['max_id'])


if __name__ == "__main__":
    get_comments('2022252207')
