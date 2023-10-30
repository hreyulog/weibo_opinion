import json


class Trust_matrix:
    def __init__(self, list_bloggers_id):
        self.dict_blogger_fans = {}
        self.dict_blogger_follows = {}
        self.all_fans_list = []
        for blogger in list_bloggers_id:
            self.dict_blogger_fans[blogger] = self.get_fans_list(blogger)
            self.dict_blogger_follows[blogger] = self.get_follow_list(blogger)
            self.all_fans_list += self.get_fans_list(blogger)

    def get_fans_list(self, user_id):
        user_fans = []
        with open('followers_crawl/' + user_id + 'fans.json', 'r', encoding='utf-8') as reader:
            for row in reader:
                json_row = json.loads(row)
                user_fans.append(json_row['id'])
        user_fans = list(set(user_fans))
        return user_fans[:3500]

    def self_trust(self, user_id):
        cont = len(self.dict_blogger_fans[user_id])
        for fan in self.dict_blogger_fans[user_id]:
            for blogger in self.dict_blogger_fans:
                if fan in self.dict_blogger_fans[blogger] and user_id != blogger:
                    cont -= 1
                    break
        return cont / len(self.dict_blogger_fans[user_id])

    def get_follow_list(self, user_id):
        user_follows = []
        with open('followers_crawl/' + user_id + 'follower.json', 'r', encoding='utf-8') as reader:
            for row in reader:
                json_row = json.loads(row)
                user_follows.append(json_row['id'])
        return user_follows

    def common_fans(self, user_id1, user_id2):
        cont = 0
        weight = 1
        for fans_user1 in self.dict_blogger_fans[user_id1]:
            if fans_user1 in self.dict_blogger_fans[user_id2]:
                cont += 1
        for follow1 in self.dict_blogger_follows[user_id1]:
            if follow1 in self.dict_blogger_follows[user_id2]:
                weight = 1.2
        return (cont / len(self.dict_blogger_fans[user_id1])) * weight

    def matrix_cal(self):
        matrix_trust = [[1 for i in range(len(self.dict_blogger_fans))] for j in range(len(self.dict_blogger_fans))]
        for i in range(len(self.dict_blogger_fans)):
            for j in range(len(self.dict_blogger_fans)):
                if i == j:
                    matrix_trust[i][j] = self.self_trust(list(self.dict_blogger_fans.keys())[i])
                else:
                    matrix_trust[i][j] = self.common_fans(list(self.dict_blogger_fans.keys())[i],
                                                          list(self.dict_blogger_fans.keys())[j])
        return matrix_trust


if __name__ == "__main__":
    list_users = []
    with open('list_bozhu.txt', 'r', encoding='utf-8') as reader:
        for row in reader:
            id = row.split()[1]
            list_users.append(id)
    matrix = Trust_matrix(list_users)
    matrix_trust = [[1 for i in range(10)] for j in range(10)]
    x = 0
    for row in matrix.matrix_cal():
        sum_row = sum(row)
        y = 0
        for i in row:
            matrix_trust[x][y] = i / sum_row
            y += 1
        x += 1
    print(matrix_trust)
