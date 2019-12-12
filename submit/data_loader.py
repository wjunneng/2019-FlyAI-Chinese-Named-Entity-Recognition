import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tokenization import BertTokenizer
from data_processor import MyPro, convert_examples_to_features
import args as args
from Logginger import init_logger

logger = init_logger("bert_ner", logging_path=args.log_path)


def init_params():
    processors = {"bert_ner": MyPro}
    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    if args.use_standard:
        tokenizer = BertTokenizer(vocab_file='./vocab.txt')
    else:
        tokenizer = BertTokenizer(vocab_file=args.VOCAB_FILE)

    return processor, tokenizer


def create_batch_iter(mode, X, y):
    """构造迭代器"""
    processor, tokenizer = init_params()
    if mode == 'train':
        examples = processor.get_train_examples(X=X, y=y)
    elif mode == 'dev':
        examples = processor.get_dev_examples(X=X, y=y)
    elif mode == 'predict':
        examples = processor.get_examples(X=X)
    else:
        raise ValueError("Invalid mode %s" % mode)
    batch_size = len(X)

    label_list = processor.get_labels()

    # 特征
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_output_mask = torch.tensor([f.output_mask for f in features], dtype=torch.long)

    # 数据集
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_output_mask)

    if mode == "train":
        sampler = RandomSampler(data)
    elif mode == "dev":
        sampler = SequentialSampler(data)
    elif mode == 'predict':
        sampler = SequentialSampler(data)
    else:
        raise ValueError("Invalid mode %s" % mode)

    # 迭代器
    iterator = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return iterator
