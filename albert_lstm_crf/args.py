import os
import sys

os.chdir(sys.path[0])

# -----------ARGS---------------------
WORDS_FILE = os.path.join(os.getcwd(), "data/input/words.dict")
VOCAB_FILE = os.path.join(os.getcwd(), 'data/input/model/vocab.txt')
data_dir = os.path.join(os.getcwd(), "data/")  # 原始数据文件夹，应包括tsv文件
output_dir = os.path.join(os.getcwd(), "data/checkpoint")  # checkpoint和预测输出文件夹
bert_model = os.path.join(os.getcwd(), "data/input/model")  # BERT 预训练模型种类 bert-base-chinese
log_path = os.path.join(os.getcwd(), "data/log")  # 日志文件

# albert_base
# embedding_size = 768
# albert_large
embedding_size = 1024
# albert_xlarge
# embedding_size = 2048

# 是否利用 伪标签 来提升模型的泛化能力
use_pseudo_labeling = True

hidden_size = 128
dropout = 0.1
learning_rate = 1e-3
task_name = "albert_lstm_crf_ner"
unknown_token = "[UNK]"
warmup_proportion = 0.1

seed = 42
max_seq_length = 128
use_calculate_max_seq_length = False

tags = ['ROLE', 'LAW', 'LOC', 'CRIME', 'TIME', 'ORG', 'PER']
token_words = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
labels = ['O', 'B-ROLE', 'B-LAW', 'B-LOC', 'B-CRIME', 'B-TIME', 'B-ORG', 'B-PER',
          'I-ROLE', 'I-LAW', 'I-LOC', 'I-CRIME', 'I-TIME', 'I-ORG', 'I-PER']
