from pickle import load, dump
from tensorflow.keras.preprocessing.text import Tokenizer


def load_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def load_imgname(filepath):
    content = load_file(filepath).split('\n')
    name_list = []
    for line in content:
        if len(line) < 1:
            continue
        name_list.append(line[:-4])  # 去掉.jpg
    return name_list


def load_descriptions(name_list, filepath):
    descriptions = {}
    content = load_file(filepath).split('\n')
    for line in content:
        sen = line.split('\t')  # 将图片id和描述分开
        img_name, description = sen[0], sen[1:]
        if img_name in name_list:
            description = 'start ' + ''.join(description) + ' end'
            if img_name not in descriptions:
                descriptions[img_name] = []
            descriptions[img_name].append(description)
    return descriptions


# 获取名单里图片的特征（加载.pkl文件）
def load_photo_features(filename, namelist):
    file = open(filename,'rb')
    all_features = load(file)
    features = {k: all_features[k] for k in namelist}   # 筛选名单内容
    file.close()
    return features


def get_all_description(descriptions):
    all_des = []
    name_list = list(descriptions.keys())
    for img_name in name_list:
        for des in list(descriptions[img_name]):
            all_des.append(des)
    return all_des


# 为图片描述创建tokenizer
def create_tokenizer(descriptions):
    lines = get_all_description(descriptions)           # 字典转列表
    tokenizer = Tokenizer()                             # 创建tokenizer
    tokenizer.fit_on_texts(lines)                       # fit
    return tokenizer


def save_tokenizer(descriptions):
    tokenizer = create_tokenizer(descriptions)
    with open('data/tokenizer.pkl', 'wb') as handle:
        dump(tokenizer, handle)
    print('tokenizer save complete')


def max_length(descriptions):
    lines = get_all_description(descriptions)
    maxlength = 0
    for line in lines:
        maxlength = max(maxlength, len(line.split()))
    return maxlength