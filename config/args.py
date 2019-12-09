import os

# -----------ARGS---------------------
ROOT_DIR = "/home/wjunneng/Ubuntu/2019-FlyAI-Chinese-Named-Entity-Recognition"
RAW_DATA = os.path.join(ROOT_DIR, "data/input/dev.csv")
STOP_WORD_LIST = None
CUSTOM_VOCAB_FILE = None
WORDS_FILE = os.path.join(ROOT_DIR, "data/input/words.dict")
VOCAB_FILE = os.path.join(ROOT_DIR, 'data/input/vocab.txt')
TRAIN = os.path.join(ROOT_DIR, "data/input/train.json")
VALID = os.path.join(ROOT_DIR, "data/input/dev.json")
log_path = os.path.join(ROOT_DIR, "output/logs")
plot_path = os.path.join(ROOT_DIR, "output/images/loss_acc.png")
data_dir = os.path.join(ROOT_DIR, "data/")  # 原始数据文件夹，应包括tsv文件
cache_dir = os.path.join(ROOT_DIR, "model/")
output_dir = os.path.join(ROOT_DIR, "output/checkpoint")  # checkpoint和预测输出文件夹

bert_model = os.path.join(ROOT_DIR, "model/pytorch_pretrained_model")  # BERT 预训练模型种类 bert-base-chinese
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
