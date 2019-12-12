# -----------ARGS---------------------
RAW_DATA = "./data/input/dev.csv"
WORDS_FILE = "./data/input/words.dict"
VOCAB_FILE = "./data/input/vocab.txt"
log_path = "./log"
data_dir = "./data/"  # 原始数据文件夹，应包括tsv文件
cache_dir = "./model/"
output_dir = "./checkpoint"  # checkpoint和预测输出文件夹
bert_model = "./data/input/model"  # BERT 预训练模型种类 bert-base-chinese

STOP_WORD_LIST = None
CUSTOM_VOCAB_FILE = None

vocab_type = "word"
# vocab_type = "words"
use_standard = True

task_name = "bert_ner"  # 训练任务名称
token_words = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
flag_words = ["[PAD]", "[CLP]", "[SEP]", "[UNK]"]
max_seq_length = 200
do_lower_case = True
learning_rate = 1e-4
warmup_proportion = 0.1
no_cuda = False
seed = 2019
gradient_accumulation_steps = 1
fp16 = False
loss_scale = 0.
labels = ['B-ROLE', 'B-LAW', 'B-LOC', 'B-CRIME', 'B-TIME', 'B-ORG', 'B-PER', 'I-PER', 'I-ORG', 'I-LOC', 'I-LAW', 'I-ROLE',
          'I-CRIME', 'I-TIME', 'O']
device = "cuda:0"
