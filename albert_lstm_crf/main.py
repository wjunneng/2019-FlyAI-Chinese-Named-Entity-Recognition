# -*- coding:utf-8 -*-
import sys
import os

os.chdir(sys.path[0])
import argparse
import torch
import math
import random
import numpy as np
from tqdm import tqdm

import shutil
from net import Net
from utils import f1_score, get_tags, format_result, convert_tf_checkpoint_to_pytorch
import args
from model_util import save_model
from data_loader import create_batch_iter
from torch.optim.adamw import AdamW

from flyai.utils import remote_helper
from flyai.dataset import Dataset
from Logginger import init_logger

logger = init_logger("bert_ner", logging_path=args.log_path)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Instructor(object):
    """
    特点：使用flyai字典的get all data  | 自己进行划分next batch
    """

    def __init__(self, args):
        self.args = args
        self.tag_map = {label: i for i, label in enumerate(self.args.labels)}

    def train(self, train_source, train_target, dev_source, dev_target):
        if os.path.exists(self.args.output_dir) is True:
            shutil.rmtree(self.args.output_dir)

        train_dataloader = create_batch_iter(mode='train', X=train_source, y=train_target, batch_size=self.args.BATCH)
        dev_dataloader = create_batch_iter(mode='dev', X=dev_source, y=dev_target, batch_size=self.args.BATCH)

        self.model.to(DEVICE)

        # 优化器准备
        param_optimizer = list(self.model.named_parameters())
        no_decay = list(['bias', 'LayerNorm.bias', 'LayerNorm.weight'])
        optimizer_grouped_parameters = list([{'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01}, {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}])

        optimizer = AdamW(params=optimizer_grouped_parameters, lr=self.args.learning_rate)

        total_size = math.ceil(len(train_source) / self.args.BATCH)

        best_acc = 0
        for epoch in range(self.args.EPOCHS):
            for train_step, train_batch in enumerate(tqdm(train_dataloader, desc='Train_Iteration')):
                self.model.train()
                self.model.zero_grad()

                train_batch = tuple(t.to(DEVICE) for t in train_batch)
                t_input_ids, t_input_mask, t_labels, t_out_masks = train_batch

                t_bert_encode = self.model(t_input_ids, t_input_mask)
                loss = self.model.loss_fn(bert_encode=t_bert_encode, tags=t_labels, output_mask=t_out_masks)
                loss.backward()

                # 梯度裁剪
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()

                if train_step % 10 == 0:
                    self.model.eval()
                    eval_loss = 0

                    for dev_step, dev_batch in enumerate(dev_dataloader):
                        dev_batch = tuple(t.to(DEVICE) for t in dev_batch)
                        d_input_ids, d_input_mask, d_label_ids, d_output_mask = dev_batch

                        with torch.no_grad():
                            d_bert_encode = self.model(d_input_ids, d_input_mask)
                        eval_loss += self.model.loss_fn(bert_encode=d_bert_encode, tags=d_label_ids,
                                                        output_mask=d_output_mask)
                        predicts = self.model.predict(d_bert_encode, d_output_mask)

                        d_label_ids = d_label_ids.view(1, -1)
                        d_label_ids = d_label_ids[d_label_ids != -1]

                        eval_acc, eval_f1 = self.model.acc_f1(predicts, d_label_ids)

                        if eval_acc > best_acc:
                            best_acc = eval_acc
                            save_model(self.model, self.args.output_dir)

                        self.model.class_report(predicts, d_label_ids)

                    logger.info("\n>step {}".format(train_step))
                    logger.info("\n>epoch [{}] {}/{}\n\tloss {:.2f}".format(epoch, train_step, total_size, loss.item()))
        if self.args.output_dir is False:
            save_model(self.model, self.args.output_dir)

    def generate(self):
        self.dataset = Dataset(epochs=self.args.EPOCHS, batch=self.args.BATCH, val_batch=self.args.BATCH)
        source, target, _, _ = self.dataset.get_all_data()
        source = np.asarray([i['source'].split(' ') for i in source])
        target = np.asarray([i['target'].split(' ') for i in target])

        index = [i for i in range(len(source))]
        np.random.shuffle(np.asarray(index))
        train_source, dev_source = source[index[0:int(len(index) * 0.9)]], source[index[int(len(index) * 0.9):]]
        train_target, dev_target = target[index[0:int(len(index) * 0.9)]], target[index[int(len(index) * 0.9):]]

        return train_source, train_target, dev_source, dev_target

    def run(self):
        # ## albert-base
        # remote_helper.get_remote_date('https://www.flyai.com/m/albert_base_zh_tensorflow.zip')
        # convert_tf_checkpoint_to_pytorch(
        #     tf_checkpoint_path="./data/input/model",
        #     bert_config_file="./data/input/model/albert_config_base.json",
        #     pytorch_dump_path="./data/input/model/pytorch_model.bin",
        #     share_type="all")

        # ## albert-large
        remote_helper.get_remote_date('https://www.flyai.com/m/albert_large_zh.zip')
        convert_tf_checkpoint_to_pytorch(
            tf_checkpoint_path="./data/input/model",
            bert_config_file="./data/input/model/albert_config_large.json",
            pytorch_dump_path="./data/input/model/pytorch_model.bin",
            share_type="all")

        # ## albert-xlarge
        # remote_helper.get_remote_date('https://www.flyai.com/m/albert_xlarge_zh_183k.zip')
        # convert_tf_checkpoint_to_pytorch(tf_checkpoint_path="./data/input/model",
        #                                  bert_config_file="./data/input/model/albert_config_xlarge.json",
        #                                  pytorch_dump_path="./data/input/model/pytorch_model.bin",
        #                                  share_type="all")

        self.model = Net(
            tag_map=self.tag_map,
            batch_size=self.args.BATCH,
            dropout=self.args.dropout,
            embedding_dim=self.args.embedding_size,
            hidden_dim=self.args.hidden_size,
        )

        train_source, train_target, dev_source, dev_target = self.generate()

        self.train(train_source, train_target, dev_source, dev_target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--EPOCHS", default=5, type=int, help="train epochs")
    parser.add_argument("-b", "--BATCH", default=4, type=int, help="batch size")
    config = parser.parse_args()

    args.EPOCHS = config.EPOCHS
    args.BATCH = config.BATCH

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    instructor = Instructor(args=args)
    instructor.run()
