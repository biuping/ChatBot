import transformers
import torch
import os
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join, exists
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split


PAD = '[PAD]'
pad_id = 0
logger = None

def get_train_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--vocab_path', default='vocabulary/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--train_raw_path', default='data/train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--train_tokenized_path', default='data/train_tokenized.txt', type=str,
                        required=False,
                        help='将原始训练语料tokenize之后的数据的存放位置')
    parser.add_argument('--log_path', default='data/training.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--raw', action='store_true', help='是否对原始训练语料做tokenize。若尚未对原始训练语料进行tokenize，则指定该参数')
    parser.add_argument('--epochs', default=10, type=int, required=False, help='训练的轮次')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--dialog_model_output_path', default='model/', type=str, required=False,
                        help='对话模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='预训练的GPT2模型的路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int, default=1, help="dataloader加载数据时使用的线程数量")
    return parser.parse_args()


def set_random_seed(hparams):
    torch.manual_seed(hparams.seed)
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)

    if hparams.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_logger(hparams):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=hparams.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def create_model(hparams, vocab_size):
    if hparams.pretrained_model:  # 如果指定了预训练的GPT2模型
        model = GPT2LMHeadModel.from_pretrained(hparams.pretrained_model)
    else:  # 若没有指定预训练模型，则初始化模型
        model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(hparams.model_config)
        model = GPT2LMHeadModel(config=model_config)
    # 根据tokenizer的vocabulary调整GPT2模型的vocab的大小
    model.resize_token_embeddings(vocab_size)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    return model, model.config.to_dict().get("n_ctx")


def preprocess_raw_data(hparams, tokenizer, n_ctx):
    '''
    对每一个轮次对话进行处理, [CLS]表示dialog开始 [SEP]表示这轮对话中的句子分隔
    形式："[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    :param hparams: 超参数
    :param tokenizer:
    :param n_ctx: GPT-2上下文窗口大小
    '''
    logger.info("tokenizing raw data,raw data path:{}, token output path:{}".format(hparams.train_raw_path,
                                                                                    hparams.train_tokenized_path))
    with open(hparams.train_raw_path, 'rb') as f:
        data = f.read().decode('utf-8')
    if "\r\n" in data:
        train_data = data.split("\r\n\r\n")
    else:
        train_data = data.split("\n\n")
    logger.info("there are {} dialog in raw dataset".format(len(train_data)))
    with open(hparams.train_tokenized_path, 'w', encoding='utf-8') as f:
        for dialog_index, dialog in enumerate(tqdm(train_data)):
            if "\r\n" in data:
                utterances = dialog.split("\r\n")
            else:
                utterances = dialog.split("\n")
            dialog_ids = [tokenizer.cls_token_id]  # [CLS]
            for utterance in utterances:
                dialog_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in utterance])
                dialog_ids.append(tokenizer.sep_token_id)  # add [SEP]
            # 截断
            dialog_ids = dialog_ids[0:n_ctx]
            for id in dialog_ids:
                f.write(str(id) + ' ')
            if dialog_index < len(train_data) - 1:
                f.write("\n")
    logger.info("finish preprocessing raw data,the result is stored in {}".format(hparams.train_tokenized_path))


def cal_loss_accuracy(outputs, labels, device):
    '''
    计算 loss 和 accuracy
    :param outputs:
    :param labels:
    :param device: 使用设备 cuda/cpu
    :return:
    '''
    logits = outputs[0]  # shape:[batch_size, token_len, vocab_size]
    # 通过前一个token预测后一个token，然后比较预测的token和label
    shift_logits = logits[..., :-1, :].contiguous()  # 深拷贝
    shift_labels = labels[..., 1:].contiguous().to(device)

    loss_fuc = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
    loss = loss_fuc(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))
    _, predicts = shift_logits.max(dim=-1)  # shape:[batch_size,token_len]

    not_ignore = shift_labels.ne(pad_id)  # 非运算，为pad_id置0，否则置1
    num_targets = not_ignore.long().sum().item()  # 非pad_id数目
    correct = (shift_labels == predicts) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets
    return loss, accuracy


def collate(batch):
    '''
    将batch所有input向其中最长input对齐
    :param batch: 一个batch_size input
    '''
    global pad_id
    input_ids = []
    batch_length = len(batch)
    max_length = 0
    for index in range(batch_length):
        if max_length < len(batch[index]):
            max_length = len(batch[index])
    # 补全
    for index in range(batch_length):
        input_len = len(batch[index])
        input_ids.append(batch[index])
        input_ids[index].extend([pad_id] * (max_length - input_len))
    return torch.tensor(input_ids, dtype=torch.long)


def train(model, device, train_list, multi_gpu, hparams):
    train_data = MyDataset(train_list)
    train_dataloader = DataLoader(train_data, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers,
                                  collate_fn=collate)
    model.train()
    total_steps = int(train_data.__len__() * hparams.epochs / hparams.batch_size / hparams.gradient_accumulation)
    logger.info('total training steps = {}'.format(total_steps))

    optimizer = transformers.AdamW(model.parameters(), lr=hparams.lr, correct_bias=True)
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=hparams.warmup_steps, t_total=total_steps)

    logger.info('starting training')
    run_loss = 0
    over_step = 0
    tb_writer = SummaryWriter(log_dir=hparams.writer_dir)
    oom_time = 0  # out of memory 次数
    for epoch in range(hparams.epochs):
        start_time = datetime.now()
        for batch_index, input_ids in enumerate(train_dataloader):
            # GPT2Model的输入为n个token_id时，输出也是n个hidden_state，使用第n个hidden_state预测第n+1个token
            input_ids = input_ids.to(device)
            try:
                outputs = model.forward(input_ids=input_ids)
                loss, accuracy = cal_loss_accuracy(outputs,input_ids, device)
                if multi_gpu:
                    loss = loss.mean()
                    accuracy = accuracy.mean()
                if hparams.gradient_accumulation > 1:
                    loss = loss/hparams.gradient_accumulation
                    accuracy = accuracy/hparams.gradient_accumulation
                loss.backward()
                # 解决梯度爆照和消失问题
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.max_grad_norm)
                if (batch_index + 1) % hparams.gradient_accumulation == 0:
                    run_loss += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()  # 清空梯度信息
                    scheduler.step()
                    over_step += 1
                    if (over_step + 1) % hparams.log_step == 0:
                        logger.info(
                            "batch {} of epoch {}, loss {}, accuracy {}".format(batch_index + 1, epoch + 1, loss,
                                                                                accuracy))
                        tb_writer.add_scalar('loss', loss.item(), over_step)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    oom_time += 1
                    logger.info("WARNING: ran out of memory,times: {}".format(oom_time))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    logger.info(str(e))
        logger.info('saving model for epoch {}'.format(epoch + 1))
        if hparams.train_mmi:
            model_path = join(hparams.mmi_model_output_path, "model_epoch{}".format(epoch+1))
        else:
            model_path = join(hparams.dialog_model_output_path, "model_epoch{}".format(epoch+1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(model_path)
        logger.info('epoch {} finished'.format(epoch + 1))
        epoch_finish_time = datetime.now()
        logger.info('time for one epoch: {}'.format(epoch_finish_time - start_time))
    logger.info('training finished')


def evaluate(model, device, test_list, multi_gpu, hparams):
    logger.info("start evaluating model")
    model.eval()
    logger.info('starting evaluating')
    tb_writer = SummaryWriter(log_dir=hparams.writer_dir)
    test_data = MyDataset(test_list)
    test_dataloader = DataLoader(test_data, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers,
                                 collate_fn=collate)
    with torch.no_grad():
        for batch_index, input_ids in enumerate(test_dataloader):
            input_ids.to(device)
            outputs = model.forward(input_ids=input_ids)
            loss, accuracy = cal_loss_accuracy(outputs, input_ids, device)

            if multi_gpu:
                loss = loss.mean()
                accuracy = accuracy.mean()
            if hparams.gradient_accumulation > 1:
                loss = loss / hparams.gradient_accumulation
                accuracy = accuracy / hparams.gradient_accumulation
            logger.info("evaluate batch {} ,loss {} ,accuracy {}".format(batch_index, loss, accuracy))
            # tb_writer.add_scalar('loss', loss.item(), overall_step)
        logger.info("finishing evaluating")


def main():
    hparams = get_train_hparams()
    global logger
    logger = create_logger(hparams)
    hparams.cuda = torch.cuda.is_available() and not hparams.no_cuda
    device = 'cuda' if hparams.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    # 设置种子
    if hparams.seed:
        set_random_seed(hparams)

    # 设置使用显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = hparams.device

    tokenizer = BertTokenizer(vocab_file=hparams.vocab_path)
    vocab_size = len(tokenizer)

    global pad_id
    pad_id = tokenizer.convert_tokens_to_ids(PAD)

    # 对话模型输出模型
    if not os.path.exists(hparams.dialog_model_output_path):
        os.mkdir(hparams.dialog_model_output_path)
    # load gpt_2 model
    model, n_ctx = create_model(hparams, vocab_size)
    model.to(device)

    if not os.path.exists(hparams.train_tokenized_path):
        preprocess_raw_data(hparams, tokenizer, n_ctx)

    multi_gpu = False  # 不使用多块gpu
    if hparams.cuda and torch.cuda.device_count() > 1:
        logger.info("Let's use GPUs to train")
        model = DataParallel(model, device_ids=[int(i) for i in hparams.device.split(',')])
        multi_gpu = True
    parameters = model.parameters()
    num_parameters = 0
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))

    # load dataset
    logger.info("loading traing data")
    with open(hparams.train_tokenized_path, "r", encoding="utf-8") as f:
        data = f.read()
    data_list = data.split("\n")
    train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=1)

    train(model, device, train_list, multi_gpu, hparams)
    evaluate(model, device, test_list, multi_gpu, hparams)


if __name__ == '__main__':
    main()