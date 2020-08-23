import transformers
import torch
import os
import argparse
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
import torch.nn.functional as F

PAD = '[PAD]'
pad_id = 0


def set_interact_hparam():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
                        help='模型参数')
    parser.add_argument('--vocab_path', default='vocabulary/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--dialog_model_path', default='model/', type=str, required=False,
                        help='dialog_model路径')
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False, help="保存聊天记录的文件路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_len', type=int, default=25, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=5, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    parser.add_argument('--batch_size', type=int, default=5, help='批量生成response，然后经过MMI模型进行筛选')
    parser.add_argument('--debug', action='store_true', help='指定该参数，可以查看生成的所有候选的reponse，及其loss')
    return parser.parse_args()

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


hparams = set_interact_hparam()
currentpath = os.path.dirname(__file__)
hparams.dialog_model_path = os.path.join(currentpath, hparams.dialog_model_path)
hparams.vocab_path = os.path.join(currentpath, hparams.vocab_path)
# 当用户使用GPU,并且GPU可用时
hparams.cuda = torch.cuda.is_available() and not hparams.no_cuda
device = 'cuda' if hparams.cuda else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = hparams.device
tokenizer = BertTokenizer(vocab_file=hparams.vocab_path)
model = GPT2LMHeadModel.from_pretrained(hparams.dialog_model_path)
model.to(device)
model.eval()


def predict(sentence, history):
    text = sentence
    history.append(tokenizer.encode(text))
    input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头

    for history_id, history_utr in enumerate(history[-hparams.max_history_len:]):
        input_ids.extend(history_utr)
        input_ids.append(tokenizer.sep_token_id)
    curr_input_tensor = torch.tensor(input_ids).long().to(device)
    generated = []
    # 最多生成max_len个token
    for _ in range(hparams.max_len):
        outputs = model(input_ids=curr_input_tensor)
        next_token_logits = outputs[0][-1, :]
        # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
        for id in set(generated):
            next_token_logits[id] /= hparams.repetition_penalty
        next_token_logits = next_token_logits / hparams.temperature
        # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
        next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=hparams.topk, top_p=hparams.topp)
        # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
            break
        generated.append(next_token.item())
        curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=0)
    history.append(generated)
    history = history[-hparams.max_history_len:]
    text = tokenizer.convert_ids_to_tokens(generated)
    reply = "".join(text)
    return reply, history


if __name__ == '__main__':
    predict()