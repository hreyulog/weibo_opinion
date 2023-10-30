from snownlp import sentiment


def get_type():
    with open('weibo_senti_100k.csv', 'r', encoding='utf-8') as reader:
        with open('online_shopping_10_cats.csv', 'r', encoding='utf-8') as reader2:
            with open('pos.txt', 'a+', encoding='utf-8') as pos_write:
                with open('neg.txt', 'a+', encoding='utf-8') as neg_write:
                    for row in reader:
                        typ, content = row.split(',')[0], row.split(',')[1].split('//')[0]
                        if typ == '0':
                            neg_write.write(content)
                        else:
                            pos_write.write(content)
                    for row in reader2:
                        cat, typ, content = row.split(',')[0], row.split(',')[1], row.split(',')[2]
                        if typ == '0':
                            neg_write.write(content)
                        else:
                            pos_write.write(content)


if __name__ == "__main__":
    # get_type()
    sentiment.train('./neg.txt', './pos.txt')
    sentiment.save('sentiment.marshal')
