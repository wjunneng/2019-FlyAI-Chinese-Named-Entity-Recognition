# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""
import argparse
from flyai.dataset import Dataset
from preprocessing.data_loader import create_batch_iter
import warnings
import time
import torch

from pytorch_pretrained_bert.optimization import BertAdam
import config.args as args
from model import Model
from net import Net
from util.plot_util import loss_acc_plot
from util.Logginger import init_logger
from util.model_util import save_model

logger = init_logger("torch", logging_path=args.log_path)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
warnings.filterwarnings('ignore')


def main():
    """
    项目的超参
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
    parser.add_argument("-b", "--BATCH", default=4, type=int, help="batch size")
    arguments = parser.parse_args()

    # ------------------判断CUDA模式----------------------
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)

    # ------------------预处理数据----------------------
    dataset = Dataset(epochs=arguments.EPOCHS, batch=arguments.BATCH)

    network = Net.from_pretrained(args.bert_model, num_tag=len(args.labels)).to(device)
    model = Model(dataset)

    # ---------------------优化器-------------------------
    param_optimizer = list(network.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    # t_total = num_train_steps

    # ---------------------GPU半精度fp16-----------------------------
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    # ------------------------GPU单精度fp32---------------------------
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion
                             # ,t_total=t_total
                             )

    # ---------------------模型初始化----------------------
    if args.fp16:
        network.half()

    network.to(device)

    train_losses = []
    eval_losses = []
    train_accuracy = []
    eval_accuracy = []

    history = {
        "train_loss": train_losses,
        "train_acc": train_accuracy,
        "eval_loss": eval_losses,
        "eval_acc": eval_accuracy
    }

    # ------------------------训练------------------------------
    best_f1 = 0
    start = time.time()
    global_step = 0
    train_loss = None
    train_acc = None
    network.train()
    for step in range(dataset.get_step()):
        x_train, y_train = dataset.next_train_batch()
        batch = create_batch_iter(mode='train', X=x_train, y=y_train).dataset.tensors
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, output_mask = batch
        bert_encode = network(input_ids, segment_ids, input_mask).cpu()

        if step == 45:
            print(1)
        train_loss = network.loss_fn(bert_encode=bert_encode, tags=label_ids, output_mask=output_mask)
        # try:
        #     train_loss = network.loss_fn(bert_encode=bert_encode, tags=label_ids, output_mask=output_mask)
        # except:
        #     continue

        if args.gradient_accumulation_steps > 1:
            train_loss = train_loss / args.gradient_accumulation_steps

        if args.fp16:
            optimizer.backward(train_loss)
        else:
            train_loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = args.learning_rate * warmup_linear(global_step, args.warmup_proportion)
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
        print("train_acc: %f" % train_acc, "train_loss: %f" % train_loss.item(), "f1: %f" % f1,
              "using time: %f" % (time.time() - start), "step: %d" % step)

        # x_val, y_val = dataset.next_validation_batch()

    # # -----------------------验证----------------------------
    # network.eval()
    # count = 0
    # y_predicts, y_labels = [], []
    # eval_loss, eval_acc, eval_f1 = 0, 0, 0
    # with torch.no_grad():
    #     for step, batch in enumerate(eval_iter):
    #         batch = tuple(t.to(device) for t in batch)
    #         input_ids, input_mask, segment_ids, label_ids, output_mask = batch
    #         bert_encode = network(input_ids, segment_ids, input_mask).cpu()
    #         eval_los = network.loss_fn(bert_encode=bert_encode, tags=label_ids, output_mask=output_mask)
    #         eval_loss = eval_los + eval_loss
    #         count += 1
    #         predicts = network.predict(bert_encode, output_mask)
    #         y_predicts.append(predicts)
    #
    #         label_ids = label_ids.view(1, -1)
    #         label_ids = label_ids[label_ids != -1]
    #         y_labels.append(label_ids)
    #
    #     eval_predicted = torch.cat(y_predicts, dim=0).cpu()
    #     eval_labeled = torch.cat(y_labels, dim=0).cpu()
    #
    #     eval_acc, eval_f1 = network.acc_f1(eval_predicted, eval_labeled)
    #     network.class_report(eval_predicted, eval_labeled)
    #
    #     logger.info(
    #         '\n\ntrain_loss: %4f - eval_loss: %4f - train_acc:%4f - eval_acc:%4f - eval_f1:%4f\n'
    #         % (train_loss.item(),
    #            eval_loss.item() / count,
    #            train_acc,
    #            eval_acc,
    #            eval_f1))
    #
    #     # 保存最好的模型
    #     if eval_f1 > best_f1:
    #         best_f1 = eval_f1
    #         save_model(network, args.output_dir)
    #
    #     train_losses.append(train_loss.item())
    #     train_accuracy.append(train_acc)
    #     eval_losses.append(eval_loss.item() / count)
    #     eval_accuracy.append(eval_acc)
    #
    # loss_acc_plot(history)

    # """
    # dataset.get_step() 获取数据的总迭代次数
    # """
    # best_score = 0
    # for step in range(dataset.get_step()):
    #     x_train, y_train = dataset.next_train_batch()
    #     x_val, y_val = dataset.next_validation_batch()
    #
    #     # model.save_model(model, MODEL_PATH, overwrite=True)
    #     print(str(step + 1) + "/" + str(dataset.get_step()))
    #
    #     train_iter, num_train_steps = create_batch_iter("train")
    #     eval_iter = create_batch_iter("dev")
    #     model = Net.from_pretrained(args.bert_model, num_tag=len(args.labels)).to(device)
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             print(name)
    #
    #     fit(model=model,
    #         training_iter=train_iter,
    #         eval_iter=eval_iter,
    #         num_epoch=args.num_train_epochs,
    #         num_train_steps=num_train_steps,
    #         verbose=1)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


if __name__ == '__main__':
    main()
