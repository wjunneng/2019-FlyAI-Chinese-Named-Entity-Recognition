# -----------ARGS---------------------
ROOT_DIR = "/home/wjunneng/Ubuntu/2019-FlyAI-Chinese-Named-Entity-Recognition"
RAW_DATA = "data/input/dev.csv"
STOP_WORD_LIST = None
CUSTOM_VOCAB_FILE = None
VOCAB_FILE = "data/input/words.dict"
TRAIN = "data/input/train.json"
VALID = "data/input/dev.json"
log_path = "output/logs"
plot_path = "output/images/loss_acc.png"
data_dir = "data/"                            # 原始数据文件夹，应包括tsv文件
cache_dir = "model/"
output_dir = "output/checkpoint"              # checkpoint和预测输出文件夹

bert_model = "model/pytorch_pretrained_model" # BERT 预训练模型种类 bert-base-chinese
task_name = "bert_ner"                      # 训练任务名称


flag_words = ["[PAD]", "[CLP]", "[SEP]", "[UNK]"]
max_seq_length = 200
do_lower_case = True
train_batch_size = 4
eval_batch_size = 4
learning_rate = 2e-5
num_train_epochs = 6
warmup_proportion = 0.1
no_cuda = False
seed = 2018
gradient_accumulation_steps = 1
fp16 = False
loss_scale = 0.
labels = ["B_PER", "I_PER", "B_T", "I_T", "B_ORG", "I_ORG", "B_LOC", "I_LOC", "O"]
device = "cuda:0"