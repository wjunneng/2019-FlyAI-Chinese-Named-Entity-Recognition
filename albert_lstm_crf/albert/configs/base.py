from pathlib import Path
import os

BASE_DIR = os.path.abspath(os.path.dirname(os.getcwd()))
BASE_DIR += '/albert_lstm_crf/albert'
config = {
    'data_dir': BASE_DIR + '/dataset/lcqmc',
    'log_dir': BASE_DIR + '/outputs/logs',
    'figure_dir': BASE_DIR + "/outputs/figure",
    'outputs': BASE_DIR + '/outputs',
    'checkpoint_dir': BASE_DIR + "/outputs/checkpoints",
    'result_dir': BASE_DIR + "/outputs/result",

    'bert_dir': '../albert_lstm_crf/data/input/model/albert_base_v2',  # 预训练模型
    'albert_config_path': '../albert_lstm_crf/data/input/model/albert_base_v2/config.json',  # 基础版的预训练模型
    'albert_vocab_path': '../albert_lstm_crf/data/input/model/albert_base_v2//vocab.txt'  # bert需要词表
    # 'bert_dir': '../albert_lstm_crf/data/input/model/albert_base_zh',  # 预训练模型
    # 'albert_config_path': '../albert_lstm_crf/data/input/model/albert_base_zh/config.json',  # 基础版的预训练模型
    # 'albert_vocab_path': '../albert_lstm_crf/data/input/model/albert_base_zh//vocab.txt'  # bert需要词表
}
