# -*- coding:utf-8 -*-

import numpy as np

import torch
from torch import nn

from albert_lstm_crf.albert.model.modeling_albert import BertConfig, BertModel
from albert_lstm_crf.albert.configs.base import config
from sklearn.metrics import f1_score, classification_report

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# import torchsnooper

def log_sum_exp(vec):
    max_score = torch.max(vec, 0)[0].unsqueeze(0)
    max_score_broadcast = max_score.expand(vec.size(1), vec.size(1))
    result = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 0)).unsqueeze(0)
    return result.squeeze(1)


class Net(nn.Module):

    def __init__(
            self,
            tag_map={'B_T': 0,
                     'I_T': 1,
                     'B_LOC': 2,
                     'I_LOC': 3,
                     'B_ORG': 4,
                     'I_ORG': 5,
                     'B_PER': 6,
                     'I_PER': 7,
                     'O': 8},
            batch_size=20,
            hidden_dim=128,
            dropout=1.0,
            embedding_dim=100
    ):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.tag_size = len(tag_map)  # 标签个数
        self.tag_map = tag_map

        bert_config = BertConfig.from_pretrained(str(config['albert_config_path']), share_type='all')
        self.word_embeddings = BertModel.from_pretrained(config['bert_dir'], config=bert_config)
        self.word_embeddings.to(DEVICE)
        self.word_embeddings.eval()

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True, dropout=self.dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.crf = CRF(self.tag_size)

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2, device=DEVICE),
                torch.randn(2, self.batch_size, self.hidden_dim // 2, device=DEVICE))

    def forward(self, input_ids, attention_mask):
        self.batch_size = input_ids.size(0)
        self.hidden = self.init_hidden()
        with torch.no_grad():
            embeddings = self.word_embeddings(input_ids=input_ids, attention_mask=attention_mask)
        # 因为在albert中的config中设置了"output_hidden_states":"True","output_attentions":"True"，所以返回所有层
        # 也可以只返回最后一层
        all_hidden_states, all_attentions = embeddings[-2:]  # 这里获取所有层的hidden_satates以及attentions
        embeddings = all_hidden_states[-2]  # 倒数第二层hidden_states的shape
        lstm_out, _ = self.lstm(embeddings, self.hidden)
        output = self.hidden2tag(lstm_out)
        return output

    def loss_fn(self, bert_encode, output_mask, tags):  # bert_encode是bert的输出
        loss = self.crf.negative_log_loss(bert_encode, output_mask, tags)
        return loss

    def predict(self, bert_encode, output_mask):
        predicts = self.crf.get_batch_best_path(bert_encode, output_mask)
        # 以下是用于主程序中的评估eval_1(); acc_f1,class_report的评估
        # predicts = predicts.view(1, -1).squeeze()
        # predicts = predicts[predicts != -1]
        return predicts

    def acc_f1(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        f1 = f1_score(y_true, y_pred, average="macro")
        correct = np.sum((y_true == y_pred).astype(int))
        acc = correct / y_pred.shape[0]
        print('acc: {}'.format(acc))
        print('f1: {}'.format(f1))
        return acc, f1

    def class_report(self, y_pred, y_true):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        classify_report = classification_report(y_true, y_pred)
        print('\n\nclassify_report:\n', classify_report)


class CRF(nn.Module):
    """
    线性条件随机场
    """

    def __init__(self, num_tag):
        if num_tag <= 0:
            raise ValueError("Invalid value of num_tag: %d" % num_tag)
        super(CRF, self).__init__()
        self.num_tag = num_tag
        self.start_tag = num_tag
        self.end_tag = num_tag + 1
        # 转移矩阵transitions：P_jk 表示从tag_j到tag_k的概率
        # P_j* 表示所有从tag_j出发的边
        # P_*k 表示所有到tag_k的边
        self.transitions = nn.Parameter(torch.Tensor(num_tag + 2, num_tag + 2))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        self.transitions.data[self.end_tag, :] = -10000  # 表示从EOS->其他标签为不可能事件, 如果发生，则产生一个极大的损失
        self.transitions.data[:, self.start_tag] = -10000  # 表示从其他标签->SOS为不可能事件, 同上

    def real_path_score(self, features, tags):
        """
        features: (seq_len, num_tag)
        tags:real tags
        real_path_score表示真实路径分数
        它由Emission score和Transition score两部分相加组成
        Emission score由LSTM输出结合真实的tag决定，表示我们希望由输出得到真实的标签
        Transition score则是crf层需要进行训练的参数，它是随机初始化的，表示标签序列前后间的约束关系（转移概率）
        Transition矩阵存储的是标签序列相互间的约束关系
        在训练的过程中，希望real_path_score最高，因为这是所有路径中最可能的路径
        """
        r = torch.LongTensor(range(features.size(0)))
        r = r.to(DEVICE)
        pad_start_tags = torch.cat([torch.LongTensor([self.start_tag]).to(DEVICE), tags])
        pad_stop_tags = torch.cat([tags, torch.LongTensor([self.end_tag]).to(DEVICE)])
        # Transition score + Emission score
        score = torch.sum(self.transitions[pad_start_tags, pad_stop_tags]) + torch.sum(features[r, tags])
        return score

    def all_possible_path_score(self, features):
        """
        features (seq_len, num_tag)
        计算所有可能的路径分数的log和：前向算法
        step1: 将forward列expand成3*3
        step2: 将下个单词的emission行expand成3*3
        step3: 将1和2和对应位置的转移矩阵相加
        step4: 更新forward，合并行
        step5: 取forward指数的对数计算total
        """
        time_steps = features.size(0)
        # 初始化
        forward = torch.zeros(self.num_tag)  # 初始化START_TAG的发射分数为0
        forward = forward.to(DEVICE)

        for i in range(0, time_steps):  # START_TAG -> 1st word -> 2nd word ->...->END_TAG
            emission_start = forward.expand(self.num_tag, self.num_tag).t()
            emission_end = features[i, :].expand(self.num_tag, self.num_tag)
            if i == 0:
                trans_score = self.transitions[self.start_tag, :self.start_tag]
            else:
                trans_score = self.transitions[:self.start_tag, :self.start_tag]
            sum = emission_start + emission_end + trans_score
            forward = log_sum(sum, dim=0)
        forward = forward + self.transitions[:self.start_tag, self.end_tag]  # END_TAG
        total_score = log_sum(forward, dim=0)
        return total_score

    def negative_log_loss(self, inputs, output_mask, tags):
        """
        inputs:(batch_size, seq_len, num_tag)
        output_mask:(batch_size, seq_len)
        tags:(batch_size, seq_len) # tags中不包括[CLS]和[SEP]
        target_function = P_real_path_score/P_all_possible_path_score
                        = exp(S_real_path_score)/ sum(exp(certain_path_score))
        我们希望P_real_path_score的概率越高越好，即target_function的值越大越好
        因此，loss_function取其相反数，越小越好，其实是负对数似然
        loss_function = -log(target_function)
                      = -S_real_path_score + log(exp(S_1 + exp(S_2) + exp(S_3) + ...))
                      = -S_real_path_score + log(all_possible_path_score)
        """
        loss = torch.tensor(0., requires_grad=True).to(DEVICE)
        num_tag = inputs.size(2)  # 序列length
        num_chars = torch.sum(output_mask.detach()).float()  # tag的token的个数
        for ix, (features, tag) in enumerate(zip(inputs, tags)):
            # 过滤[CLS] [SEP]
            # features (seq_len, num_tag)
            # output_mask (batch_size, seq_len)
            # tag: seq_len
            num_valid = torch.sum(output_mask[ix].detach())
            features = features[output_mask[ix] == 1]
            tag = tag[:num_valid]
            real_score = self.real_path_score(features, tag)
            total_score = self.all_possible_path_score(features)
            cost = total_score - real_score
            loss = loss + cost
        return loss / num_chars

    def viterbi(self, features):
        time_steps = features.size(0)
        forward = torch.zeros(self.num_tag)  # START_TAG
        forward = forward.to(DEVICE)
        # back_points 到该点的最大分数  last_points 前一个点的索引
        back_points, index_points = [self.transitions[self.start_tag, :self.start_tag].to(DEVICE)], [
            torch.LongTensor([-1]).expand_as(forward).to(DEVICE)]
        for i in range(1, time_steps):  # START_TAG -> 1st word -> 2nd word ->...->END_TAG
            emission_start = forward.expand(self.num_tag, self.num_tag).t()
            emission_end = features[i, :].expand(self.num_tag, self.num_tag)
            trans_score = self.transitions[:self.start_tag, :self.start_tag]
            sum = emission_start + emission_end + trans_score
            forward, index = torch.max(sum.detach(), dim=0)
            back_points.append(forward)
            index_points.append(index)
        back_points.append(forward + self.transitions[:self.start_tag, self.end_tag].to(DEVICE))  # END_TAG
        return back_points, index_points

    def get_best_path(self, features):
        back_points, index_points = self.viterbi(features)
        # 找到线头
        best_last_point = argmax(back_points[-1])
        index_points = torch.stack(index_points)  # 堆成矩阵
        m = index_points.size(0)
        # 初始化矩阵
        best_path = [best_last_point]
        # 循着线头找到其对应的最佳路径
        for i in range(m - 1, 0, -1):
            best_index_point = index_points[i][best_last_point]
            best_path.append(best_index_point)
            best_last_point = best_index_point
        best_path.reverse()
        return best_path

    def get_batch_best_path(self, inputs, output_mask):

        batch_best_path = []
        max_len = inputs.size(1)
        num_tag = inputs.size(2)
        for ix, features in enumerate(inputs):
            features = features[output_mask[ix] == 1]
            best_path = self.get_best_path(features)
            best_path = torch.Tensor(best_path).long()
            best_path = padding(best_path, max_len)
            batch_best_path.append(best_path)
        batch_best_path = torch.stack(batch_best_path, dim=0)
        return batch_best_path


def log_sum(matrix, dim):
    """
    前向算法是不断累积之前的结果，这样就会有个缺点
    指数和累积到一定程度后，会超过计算机浮点值的最大值，变成inf，这样取log后也是inf
    为了避免这种情况，我们做了改动：
    1. 用一个合适的值clip去提指数和的公因子，这样就不会使某项变得过大而无法计算
    SUM = log(exp(s1)+exp(s2)+...+exp(s100))
        = log{exp(clip)*[exp(s1-clip)+exp(s2-clip)+...+exp(s100-clip)]}
        = clip + log[exp(s1-clip)+exp(s2-clip)+...+exp(s100-clip)]
    where clip=max
    """
    clip_value = torch.max(matrix)  # 极大值
    clip_value = int(clip_value.data.tolist())
    log_sum_value = clip_value + torch.log(torch.sum(torch.exp(matrix - clip_value), dim=dim))
    return log_sum_value


def argmax(matrix, dim=0):
    """(0.5, 0.4, 0.3)"""
    _, index = torch.max(matrix, dim=dim)
    return index


def padding(vec, max_len, pad_token=-1):
    new_vec = torch.zeros(max_len).long()
    new_vec[:vec.size(0)] = vec
    new_vec[vec.size(0):] = pad_token
    return new_vec
