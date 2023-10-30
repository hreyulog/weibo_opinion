import csv
import json

import keras
from snownlp import SnowNLP
import re
import jieba
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors


def predict_sentiment(text, cn_model, model):
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）:]+", "", text)
    # 分词
    cut = jieba.cut(text)
    cut_list = [i for i in cut]
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.key_to_index[word]
        except KeyError:
            cut_list[i] = 0
    # padding
    tokens_pad = pad_sequences([cut_list], maxlen=116,
                               padding='pre', truncating='pre')
    # 预测
    result = model.predict(x=tokens_pad)
    return result[0][0]


def sentiment(user_id):
    dict_blog = {}
    dict_score_comment = {}
    cn_model = KeyedVectors.load_word2vec_format('sentiment_analysis/sgns.weibo.bigram',
                                                 binary=False)
    model = keras.models.load_model("sentiment_analysis/sentiment")

    with open('crawl_weibo/weibo/comment' + user_id + '.csv.json', 'r', encoding='utf-8') as reader1:
        for row in reader1:
            json_row = json.loads(row)
            comment = json_row['comment']['comment']
            comment_likes = json_row['comment']['likes']
            id = json_row['id']
            if id not in dict_blog:
                dict_blog[id] = {'content': json_row['content'], 'likes': json_row['likes'], 'time': json_row['time']}
            if comment == "":
                score_comment = 0.5
            else:
                score_comment = predict_sentiment(comment, cn_model, model)
            if id not in dict_score_comment:
                dict_score_comment[id] = [(score_comment, comment_likes)]
            else:
                dict_score_comment[id].append((score_comment, comment_likes))

    with open(user_id + 'output.json', 'w', encoding='utf-8') as writer:
        for id in dict_blog:
            writer.write(json.dumps({'id': id,
                                     'content': dict_blog[id]['content'],
                                     'content_score': predict_sentiment(comment, cn_model, model),
                                     'time': dict_blog[id]['time'],
                                     'likes': dict_blog[id]['likes'],
                                     'comment_score': dict_score_comment[id]
                                     }, ensure_ascii=False, default=float))
            writer.write('\n')


if __name__ == "__main__":
    for id in ['2561744167']:
        sentiment(id)
    # with open('list_bozhu.txt', 'r', encoding='utf-8') as reader:
    #     for row in reader:
    #         id = row.split()[1]
    #         sentiment(id)
