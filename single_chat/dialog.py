import os
import time

import hparam
import tensorflow as tf
from s2sModel import Encoder, Decoder
from data import get_data, preprocess
import sys


def generate_data(path='data/xhj.csv', sample_size=450000):
    inputs, targets = get_data(path,sample_size)
    inp_tensor, inp_token = creat_tokenize(inputs)
    targ_tensor, targ_token = creat_tokenize(targets)

    return inp_tensor, inp_token, targ_tensor, targ_token


def creat_tokenize(sentence):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_input_size, oov_token=3)
    tokenizer.fit_on_texts(sentence)

    tensor = tokenizer.texts_to_sequences(sentence)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_length_input,padding='post')

    return tensor, tokenizer


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def train_step(inputs, target, tar_token, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:   # 梯度求解
        enc_output, enc_hidden = encoder(inputs, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([tar_token.word_index['start']] * hparams.batch_size, 1)

        for t in range(1, target.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(target[:, t], predictions)

            dec_input = tf.expand_dims(target[:, t], 1)

    batch_loss = (loss / int(target.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def train(hparams):
    batch_epoch = len(input_tensor) // hparams.batch_size
    print(batch_epoch)
    encoder_hidden = encoder.initialize_hidden_state()
    checkpoint_dir = hparams.ckpt_dir
    ckpt = tf.io.gfile.listdir(checkpoint_dir)
    if ckpt:
        print("reload pretrained model")
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    BUFFER_SIZE = len(input_tensor)

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(hparams.batch_size, drop_remainder=True)
    checkpoint_dir = hparams.ckpt_dir
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_last")
    start_time = time.time()
    epoch = 0

    while epoch < hparams.epoch:
        start_time_epoch = time.time()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(batch_epoch)):
            batch_loss = train_step(inp, targ, target_token, encoder_hidden)
            total_loss += batch_loss
            print(batch_loss.numpy())

        step_time_epoch = (time.time() - start_time_epoch) / batch_epoch
        step_loss = total_loss / batch_epoch
        current_steps = +batch_epoch
        step_time_total = (time.time() - start_time) / current_steps

        print('训练总步数: {} 每步耗时: {}  最新每步耗时: {} 最新每步loss {:.4f}'.format(current_steps, step_time_total, step_time_epoch,
                                                                      step_loss.numpy()))
        if epoch == hparams.epoch:
            checkpoint.save(file_prefix=checkpoint_prefix)
        manager.save(checkpoint_number=epoch)

        sys.stdout.flush()
        epoch += 1


def predict(sentence):
    checkpoint_dir = hparams.ckpt_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    sentence = preprocess(sentence)

    inputs = [input_token.word_index.get(i, 3) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_input, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, hparams.units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_token.word_index['start']], 0)

    for t in range(max_length_target):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()

        if target_token.index_word[predicted_id] == 'end':
            break
        result += str(target_token.index_word[predicted_id]) + ' '

        dec_input = tf.expand_dims([predicted_id], 0)

    return result


hparams = hparam.generate_hparams()
vocab_input_size = hparams.enc_vocab_size
vocab_target_size = hparams.dec_vocab_size
encoder = Encoder(hparams)
decoder = Decoder(hparams)
max_length_input = 20
max_length_target = 20
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
manager = tf.train.CheckpointManager(checkpoint, directory=hparams.ckpt_dir, max_to_keep=3)
currentpath = os.path.dirname(__file__)
hparams.ckpt_dir = os.path.join(currentpath, hparams.ckpt_dir)
hparams.data_path = os.path.join(currentpath, hparams.data_path)
input_tensor, input_token, target_tensor, target_token = generate_data(hparams.data_path)

if __name__ == "__main__":
    train(hparams)

