from snownlp import SnowNLP

comment = []
with open('benchmark', 'r', encoding='utf-8') as reader:
    for row in reader:
        print(SnowNLP(row).sentiments)
