# -*- coding:utf-8 -*-
import argparse
import torch
import math
import torch.optim as optim

from net import Net
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import f1_score, get_tags, format_result, convert_tf_checkpoint_to_pytorch
import args as arguments
from model_util import save_model
from data_loader import create_batch_iter

from flyai.utils import remote_helper
from flyai.dataset import Dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class Instructor(object):
    """
    特点：使用flyai字典的get all data  | 使用提供的next_train_batch | next_validation_batch
    """

    def __init__(self, exec_type="train"):
        parser = argparse.ArgumentParser()
        parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
        parser.add_argument("-b", "--BATCH", default=24, type=int, help="batch size")
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
        else:
            self.model = None

        self.dataset = Dataset(epochs=self.epochs, batch=self.batch_size)

    def train(self):
        self.model.to(DEVICE)
        # weight decay是放在正则项（regularization）前面的一个系数，正则项一般指示模型的复杂度，
        # 所以weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大。
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0005)
        # schedule = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=100, eps=1e-4, verbose=True)
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
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()
                # schedule.step(loss)
                if step % 50 == 0:
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
                    print("-" * 50)
                    progress = ("█" * int(step * 25 / total_size)).ljust(25)
                    print("step {}".format(step))
                    print("epoch [{}] |{}| {}/{}\n\tloss {:.2f}".format(epoch, progress, step, total_size, loss.item()))

        save_model(self.model, arguments.output_dir)


if __name__ == "__main__":
    ner = Instructor("train")
    ner.train()
