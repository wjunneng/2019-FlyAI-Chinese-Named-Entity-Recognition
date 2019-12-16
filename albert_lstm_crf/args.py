# -----------ARGS---------------------
WORDS_FILE = "./data/input/words.dict"
data_dir = "./data/"  # 原始数据文件夹，应包括tsv文件
output_dir = "./checkpoint"  # checkpoint和预测输出文件夹
bert_model = "./data/input/model"  # BERT 预训练模型种类 bert-base-chinese
log_path = "./log"  # 日志文件

embedding_size = 768
hidden_size = 128
max_length = 128
dropout = 0.5
learning_rate = 1e-4
task_name = "albert_lstm_crf_ner"
unknown_token = "[UNK]"

max_seq_length = 128
use_calculate_max_seq_length = False

tags = ['ROLE', 'LAW', 'LOC', 'CRIME', 'TIME', 'ORG', 'PER']
token_words = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
labels = ['O', 'B-ROLE', 'B-LAW', 'B-LOC', 'B-CRIME', 'B-TIME', 'B-ORG', 'B-PER',
          'I-ROLE', 'I-LAW', 'I-LOC', 'I-CRIME', 'I-TIME', 'I-ORG', 'I-PER']

# STOP_WORD_LIST = None
# CUSTOM_VOCAB_FILE = None
# max_seq_length = 128
# use_calculate_max_seq_length = True
# device = "cuda:0"
# task_name = "bert_ner"  # 训练任务名称
# token_words = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
# flag_words = ["[PAD]", "[CLP]", "[SEP]", "[UNK]"]
# unknown_token = "[UNK]"
# do_lower_case = True
# learning_rate = 2e-4
# warmup_proportion = 0.4
# no_cuda = False
# seed = 2019
# gradient_accumulation_steps = 5
# fp16 = False
# loss_scale = 0.
