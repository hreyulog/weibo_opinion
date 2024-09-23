import keras
from snownlp import SnowNLP
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
import re
import jieba
import numpy as np
from matplotlib import pyplot as plt
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import pandas as pd

bert = TFBertForSequenceClassification.from_pretrained('bert_sentiment')
bert2 = TFBertForSequenceClassification.from_pretrained('bert_sentiment2')

cn_model = KeyedVectors.load_word2vec_format('sgns.weibo.bigram',
                                             binary=False)
model = keras.models.load_model("sentiment")

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)


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
    tokens_pad = pad_sequences([cut_list], maxlen=107,
                               padding='pre', truncating='pre')
    # 预测
    result = model.predict(x=tokens_pad, verbose=0)
    return result[0][0]


def tptnfpfn(predict_score, fact, threshold):
    if fact == '0':
        if predict_score < threshold:
            return 'TN'
        else:
            return 'FP'
    elif fact == '1':
        if predict_score < threshold:
            return 'FN'
        else:
            return 'TP'


def TPR(dict_type):
    return dict_type['TP'] / (dict_type['TP'] + dict_type['FN'])


def FPR(dict_type):
    return dict_type['FP'] / (dict_type['TN'] + dict_type['FP'])


def accurancy(dict_type):
    return (dict_type['TP'] + dict_type['TN']) / (dict_type['TP'] + dict_type['FP'] + dict_type['TN'] + dict_type['FN'])


def precision(dict_type):
    return (dict_type['TP'] / (dict_type['TP'] + dict_type['FP']))


def recall(dict_type):
    return (dict_type['TP'] / (dict_type['TP'] + dict_type['FN']))


def F1(p, r):
    return str(2 * p * r / (p + r))


def compute(pos_list, neg_list, threshold):
    dict_type = {'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0}
    for st in pos_list:
        dict_type[tptnfpfn(st, '1', threshold)] += 1
    for st in neg_list:
        dict_type[tptnfpfn(st, '0', threshold)] += 1
    return dict_type


def cont(type_method):
    pos_list = []
    neg_list = []
    max_row = 100
    i = 0
    with open('test.txt', 'r', encoding='utf-8') as reader:

        for row in reader:
            if i > max_row:
                break
            i += 1
            label = row.split(',')[1]
            content = row.split(',')[2]
            print(i, content)
            if label == '1':
                if type_method == 'snow':
                    pos_list.append(SnowNLP(content).sentiments)
                elif type_method == 'bilstm':
                    pos_list.append(predict_sentiment(content, cn_model, model))
                elif type_method == 'bert':
                    predict_input = tokenizer.encode(content, truncation=True, padding=True, return_tensors="tf")
                    tf_output = bert.predict(predict_input)[0]
                    tf_prediction = tf.nn.softmax(tf_output, axis=1)

                    pos_list.append(tf_prediction[0].numpy()[1])
                elif type_method == 'bert2':
                    predict_input = tokenizer.encode(content, truncation=True, padding=True, return_tensors="tf")
                    tf_output = bert2.predict(predict_input)[0]
                    tf_prediction = tf.nn.softmax(tf_output, axis=1)

                    pos_list.append(tf_prediction[0].numpy()[1])
            elif label == '0':
                if type_method == 'snow':
                    neg_list.append(SnowNLP(content).sentiments)
                elif type_method == 'bilstm':
                    neg_list.append(predict_sentiment(content, cn_model, model))
                elif type_method == 'bert':
                    predict_input = tokenizer.encode(content, truncation=True, padding=True, return_tensors="tf")
                    tf_output = bert.predict(predict_input)[0]
                    tf_prediction = tf.nn.softmax(tf_output, axis=1)

                    neg_list.append(tf_prediction[0].numpy()[1])
                elif type_method == 'bert2':
                    predict_input = tokenizer.encode(content, truncation=True, padding=True, return_tensors="tf")
                    tf_output = bert2.predict(predict_input)[0]
                    tf_prediction = tf.nn.softmax(tf_output, axis=1)

                    neg_list.append(tf_prediction[0].numpy()[1])
    return pos_list, neg_list


def roc(pos, neg):
    pred = pos + neg
    true = [1 for i in range(len(pos))] + [0 for j in range(len(neg))]
    return roc_auc_score(true, pred)


if __name__ == "__main__":
    x_snow = []
    y_snow = []
    x_bilstm = []
    y_bilstm = []
    x_bert = []
    y_bert = []
    x_bert2 = []
    y_bert2 = []
    thresholds = np.arange(0, 1, 0.1)
    print('snow')
    snow_pos, snow_neg = cont('snow')
    snow_roc = roc(snow_pos, snow_neg)
    print('bilstm')
    bilstm_pos, bilstm_neg = cont('bilstm')
    bilstm_roc = roc(bilstm_pos, bilstm_neg)
    print('bert')
    bert_pos, bert_neg = cont('bert')
    bert_roc = roc(bert_pos, bert_neg)
    print('bert2')
    bert2_pos, bert2_neg = cont('bert2')
    bert2_roc = roc(bert2_pos, bert2_neg)


    for threshold in thresholds:
        dict_type_snow = compute(snow_pos, snow_neg, threshold)
        dict_type_bilstm = compute(bilstm_pos, bilstm_neg, threshold)
        dict_type_bert = compute(bert_pos, bert_neg, threshold)
        dict_type_bert2 = compute(bert2_pos, bert2_neg, threshold)
        x_snow.append(FPR(dict_type_snow))
        y_snow.append(TPR(dict_type_snow))

        x_bert.append(FPR(dict_type_bert))
        y_bert.append(TPR(dict_type_bert))

        x_bilstm.append(FPR(dict_type_bilstm))
        y_bilstm.append(TPR(dict_type_bilstm))

        x_bert2.append(FPR(dict_type_bert2))
        y_bert2.append(TPR(dict_type_bert2))
    fig, ax = plt.subplots()
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.plot(np.array(x_snow), np.array(y_snow), label=f'naive bayes auc={round(snow_roc, 4)}')
    ax.plot(np.array(x_bilstm), np.array(y_bilstm), label=f'BiLSTM auc={round(bilstm_roc, 4)}')
    ax.plot(np.array(x_bert), np.array(y_bert), label=f'Bert2 auc={round(bert_roc, 4)}')
    ax.plot(np.array(x_bert2), np.array(y_bert2), label=f'Bert auc={round(bert2_roc, 4)}')

    dict_type_snow = compute(snow_pos, snow_neg, 0.5)
    dict_type_bilstm = compute(bilstm_pos, bilstm_neg, 0.5)
    dict_type_bert = compute(bert_pos, bert_neg, 0.5)
    dict_type_bert2 = compute(bert2_pos, bert2_neg, 0.5)

    ax.legend()
    plt.show()
    print('snow: ' + str(precision(dict_type_snow)) + ' ' + str(recall(dict_type_snow)) + ' ' + F1(
        precision(dict_type_snow),
        recall(dict_type_snow)) + ' ' + str(accurancy(dict_type_snow)))
    print(
        'bilstm: ' + str(precision(dict_type_bilstm)) + ' ' + str(recall(dict_type_bilstm)) + ' ' + F1(
            precision(dict_type_bilstm),
            recall(dict_type_bilstm)) + ' ' + str(accurancy(dict_type_bilstm)))
    print('bert2: ' + str(precision(dict_type_bert)) + ' ' + str(recall(dict_type_bert)) + ' ' + F1(
        precision(dict_type_bert),
        recall(dict_type_bert)) + ' ' + str(accurancy(dict_type_bert)))
    print('bert: ' + str(precision(dict_type_bert2)) + ' ' + str(recall(dict_type_bert2)) + ' ' + F1(
        precision(dict_type_bert2),
        recall(dict_type_bert2)) + ' ' + str(accurancy(dict_type_bert2)))


