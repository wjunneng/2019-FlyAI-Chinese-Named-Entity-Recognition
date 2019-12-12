# -*- coding: utf-8 -*
import numpy
import os
import torch
from flyai.model.base import Base
import args
from model_util import load_model, save_model
from processor import Processor
from data_loader import create_batch_iter

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

    def predict(self, **data):
        if self.net is None:
            self.net = torch.load(self.bert_model)
        x_data = self.data.predict_data(**data)
        batch = create_batch_iter(mode='predict', X=x_data, y=None).dataset.tensors
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, output_mask = batch
        bert_encode = self.net(input_ids, segment_ids, input_mask).cpu()
        predicts = self.net.predict(bert_encode, output_mask).numpy()

        return self.processor.output_y(predicts)

    def predict_all(self, datas):
        if self.net is None:
            self.net = torch.load(self.bert_model)
        labels = []
        for data in datas:
            x_data = self.data.predict_data(**data)
            batch = create_batch_iter(mode='predict', X=x_data, y=None).dataset.tensors
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, output_mask = batch
            bert_encode = self.net(input_ids, segment_ids, input_mask).cpu()
            predicts = self.net.predict(bert_encode, output_mask).numpy()
            labels.append(self.processor.output_y(predicts))

        return labels

    def save_model(self, network, path, name=None, overwrite=False):
        save_model(model=network, output_dir=path)

    # def batch_iter(self, x, y, batch_size=128):
    #     """生成批次数据"""
    #     data_len = len(x)
    #     num_batch = int((data_len - 1) / batch_size) + 1
    #
    #     indices = numpy.random.permutation(numpy.arange(data_len))
    #     x_shuffle = x[indices]
    #     y_shuffle = y[indices]
    #
    #     for i in range(num_batch):
    #         start_id = i * batch_size
    #         end_id = min((i + 1) * batch_size, data_len)
    #         yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
