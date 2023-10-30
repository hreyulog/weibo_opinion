from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import re
import jieba
# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
import warnings
import keras

warnings.filterwarnings("ignore")
cn_model = KeyedVectors.load_word2vec_format('sgns.weibo.bigram',
                                             binary=False)
train_texts_orig = []
with open('pos.txt', 'r', encoding='utf-8') as reader:
    for row in reader:
        train_texts_orig.append(row.strip())
with open('neg.txt', 'r', encoding='utf-8') as reader:
    for row in reader:
        train_texts_orig.append(row.strip())
train_tokens = []
for text in train_texts_orig:
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）:]+", "", text)
    cut = jieba.cut(text)
    cut_list = [i for i in cut]
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.key_to_index[word]
        except KeyError:
            cut_list[i] = 0
    train_tokens.append(cut_list)
num_tokens = [len(tokens) for tokens in train_tokens]
num_tokens = np.array(num_tokens)
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
print(max_tokens)
print(np.sum(num_tokens < max_tokens) / len(num_tokens))


def reverse_tokens(tokens):
    text = ''
    for i in tokens:
        if i != 0:
            text = text + cn_model.index_to_key[i]
        else:
            text = text + ' '
    return text


print(train_texts_orig[0], reverse_tokens(train_tokens[0]))
embedding_dim = 300
num_words = 195202
embedding_matrix = np.zeros((num_words, embedding_dim))
for i in range(num_words):
    embedding_matrix[i, :] = cn_model[cn_model.index_to_key[i]]
embedding_matrix = embedding_matrix.astype('float32')
train_pad = pad_sequences(train_tokens, maxlen=max_tokens,
                          padding='pre', truncating='pre')
train_pad[train_pad >= num_words] = 0
train_target = np.concatenate((np.ones(24693), np.zeros(24693)))

X_train, X_test, y_train, y_test = train_test_split(train_pad,
                                                    train_target,
                                                    test_size=0.1,
                                                    random_state=12)

model = Sequential()
model.add(Embedding(num_words,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_tokens,
                    trainable=False))
model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
model.add(LSTM(units=16, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
# 我们使用adam以0.001的learning rate进行优化
# optimizer = Adam(lr=1e-3)
# optimizer=adam_v2.Adam(learning_rate=0.0001, clipnorm=1.0, clipvalue=0.5)
optimizer = 'adam'
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
print(model.summary())

earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                 factor=0.1, min_lr=1e-5, patience=0,
                                 verbose=1)
callbacks = [
    earlystopping,
    lr_reduction
]
model.fit(X_train, y_train,
          validation_split=0.1,
          epochs=20,
          batch_size=128,
          callbacks=callbacks)
model.save('sentiment')

# model = keras.models.load_model("sentiment")
result = model.evaluate(X_test, y_test)
print('Accuracy:{0:.2%}'.format(result[1]))


def predict_sentiment(text):
    print(text)
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
    cut = jieba.cut(text)
    cut_list = [i for i in cut]
    # tokenize
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.key_to_index[word]
        except KeyError:
            cut_list[i] = 0
    # padding

    tokens_pad = pad_sequences([cut_list], maxlen=max_tokens,
                               padding='pre', truncating='pre')
    # 预测
    result = model.predict(x=tokens_pad)
    coef = result[0][0]
    if coef >= 0.5:
        print('是一例正面评价', 'output=%.2f' % coef)
    else:
        print('是一例负面评价', 'output=%.2f' % coef)


test_list = [
    '酒店设施不是新的，服务态度很不好',
    '酒店卫生条件非常不好',
    '床铺非常舒适',
    '房间很凉，不给开暖气',
    '房间很凉爽，空调冷气很足',
    '酒店环境不好，住宿体验很不好',
    '房间隔音不到位',
    '晚上回来发现没有打扫卫生',
    '因为过节所以要我临时加钱，比团购的价格贵',
    '遥遥领先说的是价格和营销技术'
]
for text in test_list:
    predict_sentiment(text)
