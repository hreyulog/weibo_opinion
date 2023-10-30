import json
import os
import random

import requests
from time import sleep

proxies = {
    'http': "http://159.138.218.175",
    'https': "http://159.138.218.175"
}


class Crawl_follow:

    def __init__(self, user_id, list_fans):
        self.table_fans = list_fans
        self.table_follow = []
        self.user_id = user_id

    # 返回随机的User-Agent
    def get_random_ua(self):
        user_agent_list = [
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1"
            "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/"
            "536.11",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
            "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
            "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",
            "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
            "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 "
            "Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
            "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24"
        ]
        return {
            "User-Agent": random.choice(user_agent_list),
            "cookie": "SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WWOd-2y5aQMBANronYeI7nO5JpX5KMhUgL.FoM0Shn7S0z0She2dJLoI7yQMJH4Ugpa97tt; SSOLoginState=1693312446; ALF=1695904446; SCF=ArwBbgYGS-iqpxaFehhbJNxigFFCyq_Ea-EVTsARvUjaYpQLNFye1LoQ_j5p2dObJjfY1XOSYMJebVrrZ8AzzwU.; SUB=_2A25J6ZXuDeRhGeFN71oR9yzPzz-IHXVrFTumrDV6PUNbktAGLWjdkW1NQA0nMZU5sPxgaH_sjfDW0sPzvqxYlO0n; _T_WM=04880d5f124d730ad5e2d22b35445f37",
        }

    # 获取内容并解析
    def get_and_parse1(self, url):
        res = requests.get(url, proxies=proxies)
        cards = res.json()['data']['cards']
        info_list = []
        try:
            for i in cards:
                if "title" not in i:
                    for j in i['card_group'][1]['users']:
                        user_name = j['screen_name']  # 用户名
                        user_id = j['id']  # 用户id
                        fans_count = j['followers_count']  # 粉丝数量
                        info = {
                            "id": user_id,
                            "name": user_name,
                            "fans_cont": fans_count,
                        }
                        info_list.append(info)
                else:
                    for j in i['card_group']:
                        user_name = j['user']['screen_name']  # 用户名
                        user_id = j['user']['id']  # 用户id
                        fans_count = j['user']['followers_count']  # 粉丝数量
                        info = {
                            "id": user_id,
                            "name": user_name,
                            "fans_cont": fans_count,
                        }
                        info_list.append(info)
            if "followers" in url:
                print("第1页关注信息爬取完毕...")
                self.table_follow += info_list
            else:
                print("第1页粉丝信息爬取完毕...")
                self.table_fans += info_list
        except Exception as e:
            print(e)

    # 爬取第一页的关注和粉丝信息
    def get_first_page(self):
        url1 = "https://m.weibo.cn/api/container/getIndex?containerid=231051_-_followers_-_" + self.user_id  # 关注
        url2 = "https://m.weibo.cn/api/container/getIndex?containerid=231051_-_fans_-_" + self.user_id  # 粉丝
        self.get_and_parse1(url1)
        self.get_and_parse1(url2)

    # 获取内容并解析
    def get_and_parse2(self, url, data):
        print(url)
        res = requests.get(url, headers=self.get_random_ua(), data=data, proxies=proxies)
        sleep(3)
        info_list = []
        if res.json()['ok'] == 0:
            return True
        if 'cards' in res.json()['data']:
            card_group = res.json()['data']['cards'][0]['card_group']
        else:
            card_group = res.json()['data']['cardlistInfo']['cards'][0]['card_group']
        for card in card_group:
            user_name = card['user']['screen_name']  # 用户名
            user_id = card['user']['id']  # 用户id
            fans_count = card['user']['followers_count']  # 粉丝数量
            info = {
                "id": user_id,
                "name": user_name,
                "fans_cont": fans_count,
            }
            print(info)
            info_list.append(info)
        if "page" in data:
            print("第{}页关注信息爬取完毕...".format(data['page']))
            self.table_follow += info_list
        else:
            print("第{}页粉丝信息爬取完毕...".format(data['since_id']))
            self.table_fans += info_list
        return False

    # 爬取关注的用户信息
    def get_follow(self, num):
        url1 = "https://m.weibo.cn/api/container/getIndex?containerid=231051_-_followers_-_{}&page={}".format(
            self.user_id,
            num)
        data1 = {
            "containerid": "231051_ - _followers_ - _" + self.user_id,
            "page": num
        }
        return self.get_and_parse2(url1, data1)

    # 爬取粉丝的用户信息
    def get_followers(self, num):
        url2 = "https://m.weibo.cn/api/container/getIndex?containerid=231051_-_fans_-_{}&since_id={}".format(
            self.user_id,
            (int(num) - 1) * 15 + 1)
        data2 = {
            "containerid": "231051_-_fans_-_" + self.user_id,
            "since_id": num
        }

        return self.get_and_parse2(url2, data2)

    def deleteDuplicate(self, li):
        delet_dup = []
        for i in li:
            if i not in delet_dup:
                delet_dup.append(i)

        return delet_dup

    def save_info(self):
        print(self.table_fans)
        set_fans = self.deleteDuplicate(self.table_fans)
        set_follow = self.deleteDuplicate(self.table_follow)
        with open(self.user_id + 'fans.json', 'w', encoding='utf-8') as writer:
            for i in set_fans:
                writer.write(json.dumps(i, ensure_ascii=False))
                writer.write('\n')
        with open(self.user_id + 'follower.json', 'w', encoding='utf-8') as writer:
            for i in set_follow:
                writer.write(json.dumps(i, ensure_ascii=False))
                writer.write('\n')


def crawl_follow(user_id):
    list_ = []
    if os.path.exists(user_id + 'fans' + '.json'):
        with open(user_id + 'fans' + '.json', 'r', encoding='utf-8') as reader:
            for row in reader:
                row_json = json.loads(row)
                list_.append(row_json)
        num_list = len(list_) // 15
        crawl = Crawl_follow(user_id, list_fans=list_)
        page = 2
        while True:
            if crawl.get_follow(page):
                break
            page += 1
        for i in range(num_list, 500):
            if crawl.get_followers(i):
                break
        crawl.save_info()
    else:
        crawl = Crawl_follow(user_id, list_fans=[])
        crawl.get_first_page()
        page = 2
        while True:
            if crawl.get_follow(page):
                break
            page += 1
        for i in range(1, 500):
            if crawl.get_followers(i):
                break
        crawl.save_info()


if __name__ == '__main__':
    # with open('../list_bozhu.txt', 'r', encoding='utf-8') as reader:
    #     for row in reader:
    #         id = row.split()[1]
    #         print(id)
    #         crawl_follow(id)
    id_list = ['5640157596']
    for id in id_list:
        crawl_follow(id)
