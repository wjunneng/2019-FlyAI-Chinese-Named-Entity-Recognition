# -----------ARGS---------------------
RAW_DATA = "./data/input/dev.csv"
WORDS_FILE = "./data/input/words.dict"
VOCAB_FILE = "./data/input/vocab.txt"
TRAIN = "./data/input/train.json"
VALID = "./data/input/dev.json"
log_path = "./output/logs"
plot_path = "./output/images/loss_acc.png"
data_dir = "./data/"  # 原始数据文件夹，应包括tsv文件
cache_dir = "./model/"
output_dir = "./output/checkpoint"  # checkpoint和预测输出文件夹
bert_model = "./model/pytorch_pretrained_model"  # BERT 预训练模型种类 bert-base-chinese

STOP_WORD_LIST = None
CUSTOM_VOCAB_FILE = None
task_name = "bert_ner"  # 训练任务名称
token_words = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
flag_words = ["[PAD]", "[CLP]", "[SEP]", "[UNK]"]
max_seq_length = 200
do_lower_case = True
train_batch_size = 4
eval_batch_size = 4
learning_rate = 2e-5
num_train_epochs = 6
warmup_proportion = 0.1
no_cuda = False
seed = 2019
gradient_accumulation_steps = 1
fp16 = False
loss_scale = 0.
labels = ['B-ROLE', 'B-LOC', 'B-CRIME', 'B-TIME', 'B-ORG', 'B-PER', 'I-PER', 'I-ORG', 'I-LOC', 'I-ROLE', 'I-CRIME',
          'I-TIME', 'O']
device = "cuda:0"
