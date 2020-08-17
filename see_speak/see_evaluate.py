# 使用bleu指标评估模型
import os

import numpy as np
from pickle import load
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
import load_func as func
import see_hparam as hparam


def get_word(index, tokenizer):
    # 从tokenizer中获得词
    for word, word_index in tokenizer.word_index.items():
        if index == word_index:
            return word
    return None


def generator_description(model, tokenizer, pictures, max_length):
    start = 'start'
    sentence = start
    for i in range(max_length):
        input_seq = tokenizer.texts_to_sequences([sentence])[0]
        input_seq = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=max_length)
        output_seq = model.predict([pictures, input_seq], verbose=0)
        output_index = np.argmax(output_seq)
        output_word = get_word(output_index, tokenizer)
        if output_word is None:
            break
        sentence = sentence + ' ' + output_word
        if output_word == 'end':
            break
    return sentence


# 使用bleu评估
def evaluate_model(model, tokenizer, pictures, description, max_length):
    y_tag, y_pre = [], []
    for img_id, des_list in description.items():
        sen_predict = generator_description(model, tokenizer, pictures[img_id], max_length)
        references = [d.split() for d in des_list]
        y_tag.append(references)
        y_pre.append(sen_predict.split())
    print('BLEU-1: %f' % corpus_bleu(y_tag, y_pre, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(y_tag, y_pre, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(y_tag, y_pre, weights=(0.34, 0.33, 0.33, 0)))
    print('BLEU-4: %f' % corpus_bleu(y_tag, y_pre, weights=(0.25, 0.25, 0.25, 0.25)))
    return None


# 获取单张图片特征
def get_picture_feature(filepath):
    model = tf.keras.applications.vgg16.VGG16()
    model.layers.pop()
    model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[-1].output)
    img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    feature = model.predict(img, verbose=0)
    return feature


def predict_single(img_path):
    tokenizer = load(open(hparams.tokenizer_path, 'rb'))
    picture = get_picture_feature(img_path)   # 获取特征
    description = generator_description(model, tokenizer, picture, hparams.max_length)
    return description[5:-3]


def main(hparams):
    name_list = func.load_imgname(hparams.train_filepath)
    print('加载训练数据完成：{}'.format(len(name_list)))
    descriptions = func.load_descriptions(name_list, hparams.des_filepath)
    print('加载训练图片的描述完成：{}'.format(len(descriptions)))
    tokenizer = func.create_tokenizer(descriptions)
    print('加载tokenizer完成：词汇量--{}，最大长度--{}'.format(len(tokenizer.word_index)+1, func.max_length(descriptions)))

    print('----------------测试---------------')
    name_list_test = func.load_imgname(hparams.test_filepath)
    print('加载测试数据完成：{}'.format(len(name_list_test)))
    description_test = func.load_descriptions(name_list_test, hparams.des_filepath)
    print('加载测试图片的描述完成：{}'.format(len(description_test)))
    features_test = func.load_photo_features(hparams.feature_filepath, name_list_test)
    print('加载测试图片的特征集完成：{}'.format(len(features_test)))
    #
    # 加载训练好的模型并评估
    evaluate_model(model, tokenizer, features_test, description_test, hparams.max_length)
    # pathlist = ['2656890977_7a9f0e4138.jpg', '2656987333_80dcc82c05.jpg','2658009523_b49d611db8.jpg']
    # for i in range(len(pathlist)):
    #     sen = predict_single('data/'+pathlist[i],model,hparams.max_length)
    #     print(sen)


hparams = hparam.get_hparams()
currentpath = os.path.dirname(__file__)
hparams.save_path = os.path.join(currentpath, hparams.save_path)
hparams.tokenizer_path = os.path.join(currentpath, hparams.tokenizer_path)
model = tf.keras.models.load_model(hparams.save_path)
# main(hparams)