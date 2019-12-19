# -*- coding:utf-8 -*-
import argparse
import torch
import math
import torch.optim as optim

from net import Net
from utils import f1_score, get_tags, format_result, convert_tf_checkpoint_to_pytorch
from albert_lstm_crf import args as arguments
from model_util import save_model
from data_loader import create_batch_iter

# 必须使用该方法下载模型，然后加载
from flyai.utils import remote_helper

from flyai.dataset import Dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

remote_helper.get_remote_date('https://www.flyai.com/m/albert_large_zh.zip')
convert_tf_checkpoint_to_pytorch(
    tf_checkpoint_path="/home/wjunneng/Ubuntu/2019-FlyAI-Chinese-Named-Entity-Recognition/albert_lstm_crf/data/input/model",
    bert_config_file="/home/wjunneng/Ubuntu/2019-FlyAI-Chinese-Named-Entity-Recognition/albert_lstm_crf/data/input/model/albert_config_large.json",
    pytorch_dump_path="/home/wjunneng/Ubuntu/2019-FlyAI-Chinese-Named-Entity-Recognition/albert_lstm_crf/data/input/model/pytorch_model.bin",
    share_type="all")


class NER(object):

    def __init__(self, exec_type="train"):
        parser = argparse.ArgumentParser()
        parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
        parser.add_argument("-b", "--BATCH", default=64, type=int, help="batch size")
        args = parser.parse_args()

        self.batch_size = args.BATCH
        self.epochs = args.EPOCHS

        self.learning_rate = arguments.learning_rate
        self.embedding_size = arguments.embedding_size
        self.hidden_size = arguments.hidden_size
        self.tags = arguments.tags
        self.dropout = arguments.dropout
        self.tag_map = {label: i for i, label in enumerate(arguments.labels)}

        if exec_type == "train":
            self.model = Net(
                tag_map=self.tag_map,
                batch_size=self.batch_size,
                dropout=self.dropout,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size,
            )
            self.restore_model()

        elif exec_type == "predict":
            self.model = Net(
                dropout=self.dropout,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size
            )
            self.restore_model()
        else:
            self.model = None

        self.dataset = Dataset(epochs=self.epochs, batch=self.batch_size)

    def restore_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path + "params.pkl"))
            print("self.model:{}".format(self.model))
            print("model restore success!")
        except Exception as error:
            print("model restore faild! {}".format(error))

    def train(self):
        self.model.to(DEVICE)
        # weight decay是放在正则项（regularization）前面的一个系数，正则项一般指示模型的复杂度，
        # 所以weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大。
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0005)
        """
        当网络的评价指标不在提升的时候，可以通过降低网络的学习率来提高网络性能:
        optimer指的是网络的优化器
        mode (str) ，可选择‘min’或者‘max’，min表示当监控量停止下降的时候，学习率将减小，max表示当监控量停止上升的时候，学习率将减小。默认值为‘min’
        factor 学习率每次降低多少，new_lr = old_lr * factor
        patience=10，容忍网路的性能不提升的次数，高于这个次数就降低学习率
        verbose（bool） - 如果为True，则为每次更新向stdout输出一条消息。 默认值：False
        threshold（float） - 测量新最佳值的阈值，仅关注重大变化。 默认值：1e-4
        cooldown(int)： 冷却时间“，当调整学习率之后，让学习率调整策略冷静一下，让模型再训练一段时间，再重启监测模式。
        min_lr(float or list):学习率下限，可为 float，或者 list，当有多个参数组时，可用 list 进行设置。
        eps(float):学习率衰减的最小值，当学习率变化小于 eps 时，则不调整学习率。
        """
        # schedule = ReduceLROnPlateau(optimizer=optimizer, mode='min',factor=0.1,patience=100,verbose=False)
        total_size = math.ceil(self.dataset.get_train_length() / self.batch_size)
        for epoch in range(self.epochs):
            for step in range(self.dataset.get_step() // self.epochs):
                self.model.train()
                # 与optimizer.zero_grad()作用一样
                self.model.zero_grad()
                x_train, y_train = self.dataset.next_train_batch()
                x_val, y_val = self.dataset.next_validation_batch()
                batch = tuple(
                    t.to(DEVICE) for t in create_batch_iter(mode='train', X=x_train, y=y_train).dataset.tensors)
                b_input_ids, b_input_mask, b_labels, b_out_masks = batch
                bert_encode = self.model(b_input_ids, b_input_mask)
                loss = self.model.loss_fn(bert_encode=bert_encode, tags=b_labels, output_mask=b_out_masks)
                loss.backward()
                # 梯度裁剪
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(),1) 
                optimizer.step()
                # schedule.step(loss)
                if step % 50 == 0:
                    self.eval_1(x_val, y_val)
                    print("-" * 50)
                    progress = ("█" * int(step * 25 / total_size)).ljust(25)
                    print("step {}".format(step))
                    print("epoch [{}] |{}| {}/{}\n\tloss {:.2f}".format(epoch, progress, step, total_size, loss.item()))

        save_model(self.model, arguments.output_dir)

    def eval_1(self, x_val, y_val):
        """
        评估所有的单个tag，如下
        Returns:

        """
        self.model.eval()
        eval_loss, eval_acc, eval_f1 = 0, 0, 0
        with torch.no_grad():
            batch = tuple(
                t.to(DEVICE) for t in create_batch_iter(mode='dev', X=x_val, y=y_val).dataset.tensors)
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, input_mask, label_ids, output_mask = batch
            bert_encode = self.model(input_ids, input_mask)
            eval_los = self.model.loss_fn(bert_encode=bert_encode, tags=label_ids, output_mask=output_mask)
            eval_loss = eval_los + eval_loss
            predicts = self.model.predict(bert_encode, output_mask)

            label_ids = label_ids.view(1, -1)
            label_ids = label_ids[label_ids != -1]

            self.model.acc_f1(predicts, label_ids)
            self.model.class_report(predicts, label_ids)
            print('eval_loss: ', eval_loss)

    def eval_2(self, x_val, y_val):
        """
        只评估:['ROLE', 'LAW', 'LOC', 'CRIME', 'TIME', 'ORG', 'PER']
        :return:
        """
        self.model.eval()
        with torch.no_grad():
            batch = create_batch_iter(mode='dev', X=x_val, y=y_val).dataset.tensors
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, input_mask, label_ids, output_mask = batch
            bert_encode = self.model(input_ids, input_mask)
            predicts = self.model.predict(bert_encode, output_mask)
            print("\teval")
            for tag in self.tags:
                f1_score(label_ids, predicts, tag, self.model.tag_map)

    """
    注意：
        1.在模型中有BN层或者dropout层时，在训练阶段和测试阶段必须显式指定train()
            和eval()。
        2.一般来说，在验证或者是测试阶段，因为只是需要跑个前向传播(forward)就足够了，
            因此不需要保存变量的梯度。保存梯度是需要额外显存或者内存进行保存的，占用了空间，
            有时候还会在验证阶段导致OOM(Out Of Memory)错误，因此我们在验证和测试阶段，最好显式地取消掉模型变量的梯度。
            使用torch.no_grad()这个上下文管理器就可以了。
    """


if __name__ == "__main__":
    ner = NER("train")
    ner.train()
