# -*- coding: utf-8 -*
import json
from flyai.processor.base import Base

from config.args import VOCAB_FILE, WORDS_FILE, token_words


class Processor(Base):
    def __init__(self):
        vocab_dict = {}
        with open(VOCAB_FILE, 'w') as file1, open(WORDS_FILE, 'r') as file2:
            for token in token_words:
                file1.write(token + '\n')
            vocab = json.load(file2).keys()
            for word in vocab:
                file1.write(word + '\n')
        with open(VOCAB_FILE, 'r') as file:
            for word in dict.fromkeys(file.readlines(), True):
                vocab_dict[word] = len(vocab_dict)
        self.vocab_dict = dict(zip(vocab_dict.values(), vocab_dict.keys()))

    def input_x(self, source):
        """
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        """
        source = source.split()

        return source

    def input_y(self, target):
        """
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        """
        target = target.split()

        return target

    def output_y(self, index):
        """
        验证时使用，把模型输出的y转为对应的结果
        """
        label = []
        for i in index:
            if i != 0:
                label.append(self.vocab_dict[i])
            else:
                break
        return label
