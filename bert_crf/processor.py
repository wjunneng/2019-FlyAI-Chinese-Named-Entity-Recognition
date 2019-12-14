# -*- coding: utf-8 -*
from flyai.processor.base import Base

import args


class Processor(Base):
    def __init__(self):
        self.label_map = {i: label for i, label in enumerate(args.labels)}

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
            label.append(self.label_map[i])
        return label
