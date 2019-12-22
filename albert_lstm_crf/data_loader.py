import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import json
from data_processor import MyPro, convert_examples_to_features
import args as args
from Logginger import init_logger
from albert.configs.base import config
from albert.model.tokenization_bert import BertTokenizer

logger = init_logger("bert_ner", logging_path=args.log_path)


def init_params():
    processors = {"albert_lstm_crf_ner": MyPro}
    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)
    processor = processors[task_name]()

    # your path for model and vocab
    VOCAB = config['albert_vocab_path']
    tokenizer = BertTokenizer.from_pretrained(VOCAB)

    return processor, tokenizer


def create_batch_iter(mode, X, y):
    """
    构造迭代器
    """
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

    # 方法一： 调整维度
    if args.use_calculate_max_seq_length:
        max_seq_length = processor._calculate_max_seq_length(X=X)
        if args.max_seq_length < max_seq_length:
            max_seq_length = args.max_seq_length

    # 方法二： 固定维度
    else:
        max_seq_length = args.max_seq_length

    # 特征
    features = convert_examples_to_features(examples=examples, max_seq_length=max_seq_length, tokenizer=tokenizer)

    all_input_ids = torch.LongTensor([f.input_ids for f in features])
    all_input_mask = torch.LongTensor([f.input_mask for f in features])
    all_label_ids = torch.LongTensor([f.label_id for f in features])
    all_output_mask = torch.LongTensor([f.output_mask for f in features])

    # 数据集
    data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_output_mask)

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
