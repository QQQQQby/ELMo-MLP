# coding: utf-8
# encoding: utf-8
import random
import re
import xml.dom.minidom
from typing import List


def read_xml(input_file) -> List[List[str]]:
    f = xml.dom.minidom.parse(input_file)
    nodes = f.getElementsByTagName('Thread')
    data = []
    l = len(nodes)
    for i in range(l):
        question = nodes[i].getElementsByTagName('RelQuestion')[0].getAttribute('RELQ_CATEGORY') + ' # '
        if nodes[i].getElementsByTagName('RelQBody')[0].firstChild:
            question += (nodes[i].getElementsByTagName('RelQSubject')[0].firstChild.data + ' ## ' +
                         nodes[i].getElementsByTagName('RelQBody')[0].firstChild.data)
        else:
            question += nodes[i].getElementsByTagName('RelQSubject')[0].firstChild.data

        answers = []
        for answer in nodes[i].getElementsByTagName('RelCText'):
            answers.append(answer.firstChild.data)
        if len(answers) == 0:
            continue

        labels = []
        for label in nodes[i].getElementsByTagName('RelComment'):
            labels.append(label.getAttribute('RELC_FACT_LABEL'))

        for j in range(len(answers)):
            # 去除长度超过200的文本
            if len(question.split()) + len(answers[j].split()) > 200:
                continue
            data.append([question, answers[j], labels[j]])
    return data


# 清洗文本数据
def cleaned(text: str) -> str:
    newText = text
    # 将大写转换成小写
    newText = newText.lower()
    # 去除换行符
    newText = re.sub(r"\n", " ", newText)
    # 去除网址
    newText = newText.split()
    for i in range(len(newText) - 1, -1, -1):
        if newText[i].startswith('http') \
                or 'www.' in newText[i] \
                or '.com' in newText[i] \
                or '.net' in newText[i] \
                or '@' in newText[i]:
            newText[i] = '.'
        if newText[i] in []:
            del newText[i]
    newText = ' '.join(newText)
    # 将不应出现的符号删除
    newText = re.sub(r"[^a-z0-9().,;!?#\'`\s]", '', newText)
    # 去除上撇号
    newText = re.sub(r"i\'m", "i am", newText)
    newText = re.sub(r"\'re", " are", newText)
    newText = re.sub(r"\'s been", " has been", newText)
    newText = re.sub(r"\'s", " is", newText)
    newText = re.sub(r"\'ve", " have", newText)

    newText = re.sub(r"\'d been", " had been", newText)
    newText = re.sub(r"\'d", " would", newText)

    newText = re.sub(r"can\'t", "cannot", newText)
    newText = re.sub(r"won\'t", "will not", newText)
    newText = re.sub(r"shan\'t", "shall not", newText)
    newText = re.sub(r"n\'t", " not", newText)

    # 在标点符号前后加上空白，并去除多余符号
    newText = re.sub(r"\.+", " . ", newText)
    newText = re.sub(r',+', ' , ', newText)
    newText = re.sub(r"\?+", ' ? ', newText)
    newText = re.sub(r"!+", ' ! ', newText)
    newText = re.sub(r";+", ' ; ', newText)
    newText = re.sub(r"\(", ' ( ', newText)
    newText = re.sub(r"\)", ' ) ', newText)
    # 去除多余的空格
    newText = newText.strip()
    newText = re.sub(r"\s{2,}", " ", newText)
    return newText


if __name__ == '__main__':
    data_types = ['train', 'dev', 'test']
    data_train = None
    data_dev = None
    data_test = None
    for data_type in data_types:
        locals()['data_' + data_type] = [[cleaned(data[0]), cleaned(data[1]), data[2]] for data in
                                         read_xml('./data/answers_' + data_type + '.xml')]
    random.shuffle(data_train)
    for data_type in data_types:
        data = locals()['data_' + data_type]
        with open('./data/answers_' + data_type + '.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(['\t'.join(d) for d in data]))
