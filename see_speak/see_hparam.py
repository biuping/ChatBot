import argparse


# 看图说话的超参数
def get_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_length', default=37, type=int, help='maximum sentence length')
    parser.add_argument('--vocab_size', default=10235, type=int)
    parser.add_argument('--train_filepath', default='data/Flickr_8k.trainImages.txt', type=str)
    parser.add_argument('--des_filepath', default='data/jieba_description.txt', type=str)
    parser.add_argument('--feature_filepath', default='data/features.pkl', type=str)
    parser.add_argument('--save_path', default='model/see_and_speak.h5', type=str)
    parser.add_argument('--test_filepath', default='data/Flickr_8k.testImages.txt', type=str)
    parser.add_argument('--tokenizer_path', default='data/tokenizer.pkl', type=str)

    parser.add_argument('--epochs', default=10, type=int)
    hparams = parser.parse_args()
    return hparams
