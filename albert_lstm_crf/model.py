# -*- coding: utf-8 -*
import torch
from flyai.model.base import Base
import args
from model_util import load_model, save_model
from processor import Processor
from data_loader import create_batch_iter
import numpy as np
from collections import Counter

__import__('net', fromlist=["Net"])


class Model(Base):
    def __init__(self, data):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = torch.device(device)
        self.data = data
        self.bert_model = args.output_dir
        self.net = load_model(self.bert_model).to(self.device)
        self.processor = Processor()

    @staticmethod
    def _split_x_data(x_data, max_seq_length):
        result_list = []
        current_list = ""
        for element in x_data[0]:
            if len(current_list + element) < max_seq_length - 2:
                current_list = current_list + element + ' '
            else:
                current_list = current_list.strip()
                result_list.append(current_list.split(' '))
                current_list = element + ' '
                continue

        current_list = current_list.split(' ')
        current_list.remove("")
        result_list.append(current_list)

        return result_list

    def predict(self, **data):
        if self.net is None:
            self.net = torch.load(self.bert_model)
        x_datas = self.data.predict_data(**data)
        x_datas = Model._split_x_data(x_datas, args.max_seq_length)
        predicts = []
        apredict = []
        print('x_datas:', x_datas)
        for x_data in x_datas:
            batch = create_batch_iter(mode='predict', X=np.array([x_data]), y=None).dataset.tensors
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, label_ids, output_mask = batch
            bert_encode = self.net(input_ids, input_mask)
            predict = self.processor.output_y(self.net.predict(bert_encode, output_mask).numpy())
            apredict.extend(predict)

            start = 0
            for index in range(len(x_data)):
                # '12' 应为1个符符
                if x_data[index].replace('.', '').isdigit() is False:
                    length = len(x_data[index])
                else:
                    length = 1

                end = start + length
                predicts.append(Counter(predict[start:end]).most_common()[0][0])
                start = end

        print('apredict:', apredict)
        print('bpredict:', predicts)
        # 规则一：剔除首个字符为I开头的情况
        predicts = self.rule_1(predicts)

        # 规则二：不能出现多个'B-PER'同时出现的情况
        predicts = self.rule_2(predicts)

        print('epredict:', predicts)
        print('\n')
        return predicts

    def rule_1(self, predicts):
        predict_0 = predicts[0]
        if predict_0.startswith('I-'):
            predicts[0] = predict_0.replace('I-', 'B-')

        return predicts

    def rule_2(self, predicts):
        predicts = ' '.join(predicts)
        # b_labels = ['B-LAW', 'B-CRIME', 'B-TIME', 'B-ORG', 'B-PER']
        b_labels = ['B-ORG']
        for b_label in b_labels:
            in_end = predicts.split(b_label + ' ' + b_label)
            in_start = in_end[0].strip()
            in_end = in_end[1:]

            index = 0
            for step in in_end:
                step = step.strip()
                if step.startswith(b_label) and index != 0:
                    in_start = in_start + ' ' + b_label + ' I' + b_label[1:] + ' I' + step[1:]
                else:
                    index += 1
                    if in_start != '':
                        in_start = in_start + ' ' + b_label + ' I' + b_label[1:] + ' ' + step
                    else:
                        in_start = b_label + ' I' + b_label[1:] + ' ' + step
            predicts = in_start

        return predicts.split(' ')

    def predict_all(self, datas):
        if self.net is None:
            self.net = torch.load(self.bert_model)
        labels = []
        for data in datas:
            predicts = self.predict(source=data['source'])

            labels.append(predicts)

        return labels

    def save_model(self, network, path, name=None, overwrite=False):
        save_model(model=network, output_dir=path)
