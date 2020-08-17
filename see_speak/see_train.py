import tensorflow as tf
import numpy as np
import load_func
import see_hparam


def generate_seq(tokenizer, des_list, pictures, hparams):
    # des_list 图片描述
    X1, X2, y = [], [], []  # X1:图片特征， X2：图片描述（上文） y：标签
    for line in des_list:
        seq = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(seq)):
            seq_in, seq_out = seq[:i], seq[i]
            seq_in = tf.keras.preprocessing.sequence.pad_sequences([seq_in], maxlen=hparams.max_length)[0]
            # 转one-hot
            seq_out = tf.keras.utils.to_categorical([seq_out], num_classes=hparams.vocab_size)[0]
            X1.append(pictures)
            X2.append(seq_in)
            y.append(seq_out)
    return np.array(X1), np.array(X2), np.array(y)


def build_model(hparams):
    input1 = tf.keras.layers.Input(shape=(1000,))
    feature1 = tf.keras.layers.Dropout(0.5)(input1)
    feature2 = tf.keras.layers.Dense(256, activation='relu')(feature1)
    # feature3 = tf.keras.layers.RepeatVector(hparams.max_length)(feature2)

    input2 = tf.keras.layers.Input(shape=(hparams.max_length,))
    embedding = tf.keras.layers.Embedding(hparams.vocab_size, 256, mask_zero=True)(input2)
    se2 = tf.keras.layers.Dropout(0.5)(embedding)
    lstm = tf.keras.layers.LSTM(256)(se2)

    decoder1 = tf.keras.layers.concatenate([feature2, lstm])
    # lstm = tf.keras.layers.LSTM(500, return_sequences=False)(merged)
    # outputs = tf.keras.layers.Dense(hparams.vocab_size, activation='softmax')(lstm)
    decoder2 = tf.keras.layers.Dense(256, activation='relu')(decoder1)
    outputs = tf.keras.layers.Dense(hparams.vocab_size, activation='softmax')(decoder2)

    model = tf.keras.models.Model(inputs=[input1, input2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def data_generator(descriptions, pictures, tokenizer, hparams):
    while True:
        '''
            descriptions:{图片id，[对图片的描述]}
        '''
        for img_id, des_list in descriptions.items():
            picture = pictures[img_id][0]
            img_input, seq_input, word_out = generate_seq(tokenizer, des_list, picture, hparams)
            # back = [[img_input, seq_input], word_out]
            yield ([img_input, seq_input], word_out)


def categorical_crossentropy_from_logits(y_true, y_pred):
    y_true = y_true[:, :-1, :]  # Discard the last timestep
    y_pred = y_pred[:, :-1, :]  # Discard the last timestep
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                 logits=y_pred)
    return loss


def categorical_accuracy_with_variable_timestep(y_true, y_pred):
    y_true = y_true[:, :-1, :]  # Discard the last timestep
    y_pred = y_pred[:, :-1, :]  # Discard the last timestep

    # Flatten the timestep dimension
    shape = tf.shape(y_true)
    y_true = tf.reshape(y_true, [-1, shape[-1]])
    y_pred = tf.reshape(y_pred, [-1, shape[-1]])

    # Discard rows that are all zeros as they represent padding words.
    is_zero_y_true = tf.equal(y_true, 0)
    is_zero_row_y_true = tf.reduce_all(is_zero_y_true, axis=-1)
    y_true = tf.boolean_mask(y_true, ~is_zero_row_y_true)
    y_pred = tf.boolean_mask(y_pred, ~is_zero_row_y_true)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=1),
                                              tf.argmax(y_pred, axis=1)),
                                    dtype=tf.float32))
    return accuracy


def train(hparams):
    name_list = load_func.load_imgname(hparams.train_filepath)
    descriptions = load_func.load_descriptions(name_list, hparams.des_filepath)
    features = load_func.load_photo_features(hparams.feature_filepath, name_list)
    tokenizer = load_func.create_tokenizer(descriptions)
    model = build_model(hparams)
    steps = len(descriptions)
    for i in range(hparams.epochs):
        generator = data_generator(descriptions, features, tokenizer, hparams)
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        if i == hparams.epochs - 1:
            model.save('see_model/see_and_speak.h5')
        else:
            model.save('see_model/see_and_speak_'+str(i)+'.h5')


if __name__ == '__main__':
    hparams = see_hparam.get_hparams()
    train(hparams)

