# -----------ARGS---------------------
WORDS_FILE = "./data/input/words.dict"
VOCAB_FILE = "./data/input/vocab.txt"
STAND_VOCAB_FILE = './vocab.txt'
log_path = "./log"
data_dir = "./data/"  # 原始数据文件夹，应包括tsv文件
output_dir = "./checkpoint"  # checkpoint和预测输出文件夹
bert_model = "./data/input/model"  # BERT 预训练模型种类 bert-base-chinese

STOP_WORD_LIST = None
CUSTOM_VOCAB_FILE = None

# 第一种 -e=10 -b=20 score:32
# max_seq_length = 256
# vocab_type = "word"
# use_standard = True

# 第二种 e=10 -b=20 score: 10
# max_seq_length = 256
# vocab_type = "word"
# use_standard = False
# use_calculate_max_seq_length = True

# 第三种
# max_seq_length = 128
# vocab_type = "words"
# use_standard = False

# 第四种
# max_seq_length = 256
# vocab_type = "words"
# use_standard = True

task_name = "bert_ner"  # 训练任务名称
token_words = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
flag_words = ["[PAD]", "[CLP]", "[SEP]", "[UNK]"]
do_lower_case = True
learning_rate = 1e-4
warmup_proportion = 0.4
no_cuda = False
seed = 2019
gradient_accumulation_steps = 1
fp16 = False
loss_scale = 0.
labels = ['B-ROLE', 'B-LAW', 'B-LOC', 'B-CRIME', 'B-TIME', 'B-ORG', 'B-PER', 'I-PER', 'I-ORG', 'I-LOC', 'I-LAW',
          'I-ROLE', 'I-CRIME', 'I-TIME', 'O']
device = "cuda:0"
