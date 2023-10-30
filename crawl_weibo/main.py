import json
from get_comments import get_comments
from opinion_dynamic.sentiment import sentiment

from weiboSpider.weibo_spider.spider import Spider


class opinion:
    def __init__(self, user_id):
        self.user_id = user_id
        with open('config.json') as f:
            self.config = json.loads(f.read())
        self.config[
            'cookie'] = "_T_WM=36606812338; SCF=ArwBbgYGS-iqpxaFehhbJNxigFFCyq_Ea-EVTsARvUjamxWZMvR_tNNeaH6gFzVzeCPp7dDT1AK3GtEIQQJIbd4.; SUB=_2A25IP8VvDeRhGeFN71oR9yzPzz-IHXVrNVinrDV6PUNbktANLWL-kW1NQA0nMRNJLvwgHJsV07Sz1dVNmqmwMx_9; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WWOd-2y5aQMBANronYeI7nO5JpX5KMhUgL.FoM0Shn7S0z0She2dJLoI7yQMJH4Ugpa97tt; SSOLoginState=1698411839; ALF=1701003839; MLOGIN=1; M_WEIBOCN_PARAMS=luicode%3D20000174"
        self.config['user_id_list'] = [user_id]
        self.config['since_date'] = "2023-02-02"
        self.config['write_mode'] = ["csv"]
        self.crawl_blogs()
        get_comments(user_id)
        # sentiment(user_id)

    def crawl_blogs(self):
        wb = Spider(self.config)
        wb.start(0)


if __name__ == "__main__":
    #2245266941
    #2561744167
    for id in ['2245266941']:
        opinion(id)
    # with open('list_crawler.txt', 'r', encoding='utf-8') as reader:
    #     for row in reader:
    #         id = row.split()[1]
    #         Opinion = opinion(id)
