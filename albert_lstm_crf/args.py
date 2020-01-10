# -----------ARGS---------------------
WORDS_FILE = "./data/input/words.dict"
VOCAB_FILE = './data/input/model/vocab.txt'
data_dir = "./data/"  # 原始数据文件夹，应包括tsv文件
output_dir = "./checkpoint"  # checkpoint和预测输出文件夹
bert_model = "./data/input/model"  # BERT 预训练模型种类 bert-base-chinese
log_path = "./data/log"  # 日志文件

# albert_base
# embedding_size = 768
# albert_large
embedding_size = 1024
# albert_xlarge
# embedding_size = 2048

hidden_size = 128
dropout = 0.1
learning_rate = 1e-3
task_name = "albert_lstm_crf_ner"
unknown_token = "[UNK]"

max_seq_length = 128
use_calculate_max_seq_length = False

tags = ['ROLE', 'LAW', 'LOC', 'CRIME', 'TIME', 'ORG', 'PER']
token_words = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
labels = ['O', 'B-ROLE', 'B-LAW', 'B-LOC', 'B-CRIME', 'B-TIME', 'B-ORG', 'B-PER',
          'I-ROLE', 'I-LAW', 'I-LOC', 'I-CRIME', 'I-TIME', 'I-ORG', 'I-PER']
