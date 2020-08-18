import pandas as pd
import numpy as np


def preprocess(sentence):
    sentence = str(sentence).strip()
    sentence = 'start ' + sentence + ' end'
    return sentence


def get_data(filepath='data/xhj.csv', maxlength=450000):
    df = pd.read_csv(filepath, encoding='utf-8')
    conversations = np.array(df)[:maxlength]
    questions = [preprocess(c[0]) for c in conversations]
    answers = [preprocess(c[1]) for c in conversations]
    return tuple(questions), tuple(answers)

