# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""
import os
import argparse
import warnings
import time
import torch
from flyai.dataset import Dataset
from flyai.utils import remote_helper

from Logginger import init_logger
from data_loader import create_batch_iter
from optimization import BertAdam
import args as arguments
from net import Net
from model_util import save_model

logger = init_logger("torch", logging_path=arguments.log_path)

torch.manual_seed(arguments.seed)
torch.cuda.manual_seed(arguments.seed)
torch.cuda.manual_seed_all(arguments.seed)
warnings.filterwarnings('ignore')

remote_helper.get_remote_date("https://www.flyai.com/m/chinese_base.zip")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main():
    """
    项目的超参
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--EPOCHS", default=50, type=int, help="train epochs")
    parser.add_argument("-b", "--BATCH", default=8, type=int, help="batch size")
    args = parser.parse_args()

    # ------------------判断CUDA模式----------------------
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)

    # ------------------预处理数据----------------------
    dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)

    network = Net.from_pretrained(arguments.bert_model, num_tag=len(arguments.labels)).to(device)
    logger.info('\n预处理结束！！！\n')
    # ---------------------优化器-------------------------
    param_optimizer = list(network.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    t_total = int(dataset.get_train_length() / arguments.gradient_accumulation_steps / args.BATCH * args.EPOCHS)

    # ---------------------GPU半精度fp16-----------------------------
    if arguments.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=arguments.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if arguments.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=arguments.loss_scale)

    # ------------------------GPU单精度fp32---------------------------
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=arguments.learning_rate,
                             warmup=arguments.warmup_proportion,
                             t_total=t_total
                             )

    # ---------------------模型初始化----------------------
    if arguments.fp16:
        network.half()

    train_losses = []
    eval_losses = []
    train_accuracy = []
    eval_accuracy = []

    best_f1 = 0
    start = time.time()
    global_step = 0
    for e in range(args.EPOCHS):
        network.train()
        for step in range(dataset.get_step() // args.EPOCHS):
            x_train, y_train = dataset.next_train_batch()
            batch = create_batch_iter(mode='train', X=x_train, y=y_train).dataset.tensors
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, output_mask = batch
            bert_encode = network(input_ids, segment_ids, input_mask)
            train_loss = network.loss_fn(bert_encode=bert_encode, tags=label_ids, output_mask=output_mask)

            if arguments.gradient_accumulation_steps > 1:
                train_loss = train_loss / arguments.gradient_accumulation_steps

            if arguments.fp16:
                optimizer.backward(train_loss)
            else:
                train_loss.backward()

            if (step + 1) % arguments.gradient_accumulation_steps == 0:
                def warmup_linear(x, warmup=0.002):
                    if x < warmup:
                        return x / warmup
                    return 1.0 - x

                # modify learning rate with special warm up BERT uses
                lr_this_step = arguments.learning_rate * warmup_linear(global_step / t_total,
                                                                       arguments.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            predicts = network.predict(bert_encode, output_mask)
            label_ids = label_ids.view(1, -1)
            label_ids = label_ids[label_ids != -1]
            label_ids = label_ids.cpu()

            train_acc, f1 = network.acc_f1(predicts, label_ids)

        logger.info("\n train_acc: %f - train_loss: %f - f1: %f - using time: %f - step: %d \n" % (train_acc,
                                                                                                   train_loss.item(),
                                                                                                   f1,
                                                                                                   (
                                                                                                           time.time() - start),
                                                                                                   step))

        # -----------------------验证----------------------------
        network.eval()
        count = 0
        y_predicts, y_labels = [], []
        eval_loss, eval_acc, eval_f1 = 0, 0, 0
        with torch.no_grad():
            for step in range(dataset.get_step() // args.EPOCHS):
                x_val, y_val = dataset.next_validation_batch()
                batch = create_batch_iter(mode='dev', X=x_val, y=y_val).dataset.tensors
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, output_mask = batch
                bert_encode = network(input_ids, segment_ids, input_mask).cpu()
                eval_los = network.loss_fn(bert_encode=bert_encode, tags=label_ids, output_mask=output_mask)
                eval_loss = eval_los + eval_loss
                count += 1
                predicts = network.predict(bert_encode, output_mask)
                y_predicts.append(predicts)

                label_ids = label_ids.view(1, -1)
                label_ids = label_ids[label_ids != -1]
                y_labels.append(label_ids)

            eval_predicted = torch.cat(y_predicts, dim=0).cpu()
            eval_labeled = torch.cat(y_labels, dim=0).cpu()
            print('eval:')
            print(eval_predicted.numpy().tolist())
            print(eval_labeled.numpy().tolist())

            eval_acc, eval_f1 = network.acc_f1(eval_predicted, eval_labeled)
            network.class_report(eval_predicted, eval_labeled)

            logger.info(
                '\n\nEpoch %d - train_loss: %4f - eval_loss: %4f - train_acc:%4f - eval_acc:%4f - eval_f1:%4f\n'
                % (e + 1, train_loss.item(), eval_loss.item() / count, train_acc, eval_acc, eval_f1))

            # 保存最好的模型
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                save_model(network, arguments.output_dir)

            if e % 1 == 0:
                train_losses.append(train_loss.item())
                train_accuracy.append(train_acc)
                eval_losses.append(eval_loss.item() / count)
                eval_accuracy.append(eval_acc)


if __name__ == '__main__':
    main()
