import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from os import listdir
from pickle import dump


# 提取图片特征
def get_features(dir):
    model = VGG16()
    model.layers.pop()  # 去掉最后一层
    model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[-1].output)
    model.summary()

    features = {}
    img_name_list = listdir(dir)
    for img_name in img_name_list:
        name = dir + '/' +img_name
        print("处理图片{}".format(img_name))
        try:
            img = load_img(name, target_size=(224, 224))
        except(OSError, NameError):
            print('OSError, Path:', img_name)
            continue
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        feature = model.predict(img, verbose=0)
        features[img_name[:-4]] = feature
    return features


if __name__ == '__main__':
    directory = 'data'
    features = get_features(directory)
    dump(features, open('features.pkl', 'wb'))  # 保存文件
    print('图片特征提取完成，文件已保存！')

