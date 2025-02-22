# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert BERT checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch

from model.modeling_albert import BertConfig, BertForPreTraining, load_tf_weights_in_albert

import logging

logging.basicConfig(level=logging.INFO)


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, share_type, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_pretrained(bert_config_file, share_type=share_type)
    # print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_albert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--tf_checkpoint_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to the TensorFlow checkpoint path.")
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument('--share_type',
                        default='all',
                        type=str,
                        choices=['all', 'attention', 'ffn', 'None'])
    parser.add_argument("--pytorch_dump_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path,
                                     args.bert_config_file,
                                     args.share_type,
                                     args.pytorch_dump_path)

"""
example:

python convert_albert_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path=./pretrain/tf/albert_xlarge_zh \
    --bert_config_file=./configs/albert_config_xlarge.json \
    --pytorch_dump_path=./pretrain/pytorch/albert_xlarge_zh/pytorch_model.bin \
    --share_type=all
    
python convert_albert_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path=/home/wjunneng/Ubuntu/NLP/nlp/albert_lstm_crf_ner/src/albert/pretrain/pytorch/albert_base_zh \
    --bert_config_file=/home/wjunneng/Ubuntu/NLP/nlp/albert_lstm_crf_ner/src/albert/pretrain/pytorch/albert_base_zh/albert_config_base.json \
    --pytorch_dump_path=/home/wjunneng/Ubuntu/NLP/nlp/albert_lstm_crf_ner/src/albert/pretrain/pytorch/albert_base_zh/pytorch_model.bin \
    --share_type=all

"""
