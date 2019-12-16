# -*- coding: utf-8 -*
import torch
from flyai.model.base import Base
from model_util import load_model, save_model
from processor import Processor
from data_loader import create_batch_iter
import numpy as np
from bert_crf import args

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
        for x_data in x_datas:
            batch = create_batch_iter(mode='predict', X=np.array([x_data]), y=None).dataset.tensors
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, output_mask = batch
            bert_encode = self.net(input_ids, segment_ids, input_mask).cpu()
            predicts.extend(self.processor.output_y(self.net.predict(bert_encode, output_mask).numpy()))

        return predicts[1:]

    def predict_all(self, datas):
        if self.net is None:
            self.net = torch.load(self.bert_model)
        labels = []
        for data in datas:
            x_data = self.data.predict_data(**data)
            x_datas = Model._split_x_data(x_data, args.max_seq_length)
            predicts = []
            for x_data in x_datas:
                batch = create_batch_iter(mode='predict', X=np.array([x_data]), y=None).dataset.tensors
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, output_mask = batch
                bert_encode = self.net(input_ids, segment_ids, input_mask).cpu()
                predicts.extend(self.processor.output_y(self.net.predict(bert_encode, output_mask).numpy()))

            labels.append(predicts[1:])

        return labels

    def save_model(self, network, path, name=None, overwrite=False):
        save_model(model=network, output_dir=path)
