# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""

import argparse
import torch
from flyai.dataset import Dataset
from model import Model
from net import Net
from path import MODEL_PATH
from preprocessing.data_loader import create_batch_iter
from train import fit
from util.porgress_util import ProgressBar

"""
样例代码仅供参考学习，可以自己修改实现逻辑。
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
"""


def main():
    """
    项目的超参
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
    parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
    args = parser.parse_args()

    # 判断gpu是否可用
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)

    # 预处理数据
    dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
    model = Model(dataset)

    train_iter, num_train_steps = create_batch_iter("train")
    eval_iter = create_batch_iter("dev")
    epoch_size = num_train_steps * args.train_batch_size * args.gradient_accumulation_steps / args.num_train_epochs
    pbar = ProgressBar(epoch_size=epoch_size, batch_size=args.train_batch_size)
    model = Net.from_pretrained(args.bert_model, num_tag=len(args.labels)).to(device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    fit(model=model,
        training_iter=train_iter,
        eval_iter=eval_iter,
        num_epoch=args.num_train_epochs,
        pbar=pbar,
        num_train_steps=num_train_steps,
        verbose=1)

    """
    dataset.get_step() 获取数据的总迭代次数
    """
    best_score = 0
    for step in range(dataset.get_step()):
        x_train, y_train = dataset.next_train_batch()
        x_val, y_val = dataset.next_validation_batch()

        # model.save_model(model, MODEL_PATH, overwrite=True)
        print(str(step + 1) + "/" + str(dataset.get_step()))


if __name__ == '__main__':
    main()
