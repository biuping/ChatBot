# -*- coding: utf-8 -*-
# 准备文本数据（图片描述）
# 加载原始txt数据，清洗后保存
import string
import jieba
import load_func as func
import re

# 清洗文本
def clean_dsc(dsc):
    dsc = [word.lower() for word in dsc]           # 转小写
    dsc = [word for word in dsc if len(word)>1]    # 去除单字符
    dsc = ' '.join(dsc)                            # list2str
    for i in (string.punctuation + string.digits):
        dsc = dsc.replace(i,'')                    # 去除字符、数字 (单词内部的)
    return dsc

# 保存txt文件
def save_descriptions(txt, filename):
    txt = txt.split('\n')                          # 按换行符分割
    descriptions = []
    for line in txt:
        line = line.split()                        # 再次分割
        if len(line)<2:
            continue
        img_id, img_dsc = line[0][:-10], line[1:]   # 提取图片名和图片描述
        img_dsc = clean_dsc(img_dsc)               # 清洗描述文本
        img_dsc = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "",img_dsc)
        img_dsc = " ".join(jieba.cut(img_dsc))
        descriptions.append(img_id + '\t' + img_dsc) # 拼接图片名和图片描述
    data = '\n'.join(descriptions)                 # 每条描述间用换行符分隔
    file = open(filename, 'w')
    file.write(data)
    file.close()
    return print('Descriptions saved!')

filename = 'data/ch.caption.txt'
txt = func.load_doc(filename)
save_descriptions(txt, 'jieba_description.txt')