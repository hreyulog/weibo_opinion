import csv
import json
import re

import requests

user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36'
headers = {'User_Agent': user_agent,
           'cookie': "SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WWOd-2y5aQMBANronYeI7nO5JpX5KMhUgL.FoM0Shn7S0z0She2dJLoI7yQMJH4Ugpa97tt; SSOLoginState=1693312446; ALF=1695904446; SCF=ArwBbgYGS-iqpxaFehhbJNxigFFCyq_Ea-EVTsARvUjaYpQLNFye1LoQ_j5p2dObJjfY1XOSYMJebVrrZ8AzzwU.; SUB=_2A25J6ZXuDeRhGeFN71oR9yzPzz-IHXVrFTumrDV6PUNbktAGLWjdkW1NQA0nMZU5sPxgaH_sjfDW0sPzvqxYlO0n; _T_WM=04880d5f124d730ad5e2d22b35445f37"}


def main():
    uid='6048569942'
    uid_str = "230283" + str(uid)
    url = "https://m.weibo.cn/api/container/getIndex?containerid={}_-_INFO&title=%E5%9F%BA%E6%9C%AC%E8%B5%84%E6%96%99&luicode=10000011&lfid={}&featurecode=10000326".format(
        uid_str, uid_str)
    data = {
        "containerid": "{}_-_INFO".format(uid_str),
        "title": "基本资料",
        "luicode": 10000011,
        "lfid": int(uid_str),
        "featurecode": 10000326
    }


if __name__ == "__main__":
    main()
