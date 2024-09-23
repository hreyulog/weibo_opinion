from snownlp import SnowNLP

comment = []
cont = 0
right = 0
TP = 0
TN = 0
FP = 0
FN = 0
with open('neg_test.txt', 'r', encoding='utf-8') as reader:
    for row in reader:
        cont += 1
        if SnowNLP(row).sentiments < 0.5:
            right+=1
            TN += 1
        else:
            FP += 1
with open('pos_test.txt', 'r', encoding='utf-8') as reader:
    for row in reader:
        cont += 1
        if SnowNLP(row).sentiments < 0.5:
            FN += 1
        else:
            right += 1
            TP += 1

print((TP + TN) / (TP + TN + FP + FN))
