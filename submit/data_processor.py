import numpy as np
from Logginger import init_logger
import args as args

logger = init_logger("bert_ner", logging_path=args.log_path)


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """创建一个输入实例
        Args:
            guid: 每个example拥有唯一的id
            text_a: 第一个句子的原始文本，一般对于文本分类来说，只需要text_a
            text_b: 第二个句子的原始文本，在句子对的任务中才有，分类问题中为None
            label: example对应的标签，对于训练集和验证集应非None，测试集为None
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeature(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, output_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.output_mask = output_mask


class DataProcessor(object):
    """数据预处理的基类，自定义的MyPro继承该类"""

    def get_train_examples(self, X, y):
        """读取训练集 Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, X, y):
        """读取验证集 Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_examples(self, X):
        """读取单条数据"""
        raise NotImplementedError()

    def get_labels(self):
        """读取标签 Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, X, y):
        lines = []
        for source, target in zip(X, y):
            lines.append({"source": source, "target": target})

        return lines


class MyPro(DataProcessor):
    """将数据构造成example格式"""

    def _create_example(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            guid = "%s-%d" % (set_type, i)
            text_a = line["source"].tolist()
            label = ' '.join(line["target"].tolist())
            example = InputExample(guid=guid, text_a=text_a, label=label)
            examples.append(example)
        return examples

    def _calculate_max_seq_length(self, X):
        max_seq_length = 0
        for i in X:
            current_seq_length = 0
            for j in i:
                current_seq_length += len(j)
            if current_seq_length > max_seq_length:
                max_seq_length = current_seq_length

        return max_seq_length + 2

    def get_train_examples(self, X, y):
        lines = self._read_data(X=X, y=y)
        examples = self._create_example(lines, "train")
        return examples

    def get_dev_examples(self, X, y):
        lines = self._read_data(X=X, y=y)
        examples = self._create_example(lines, "dev")
        return examples

    def get_examples(self, X):
        examples = []
        for i, line in enumerate(X):
            guid = "predict-%d" % i
            text_a = ' '.join(line.tolist())
            label = None
            example = InputExample(guid=guid, text_a=text_a, label=label)
            examples.append(example)
        return examples

    def get_labels(self):
        return args.labels


def convert_tokens_to_ids(tokenizer, tokens):
    result = []
    for token in tokens:
        if token in tokenizer:
            result.append(tokenizer.index(token))
        else:
            result.append(tokenizer.index(args.unknown_token))

    return result


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    # 标签转换为数字
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    labels = None
    for ex_index, example in enumerate(examples):
        tokens_a = example.text_a
        if example.label is not None:
            labels = example.label.split()

        if len(tokens_a) == 0:
            continue

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
            if labels is not None:
                labels = labels[:(max_seq_length - 2)]

        # ----------------处理source--------------
        # 句子首尾加入标示符
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        # 词转换成数字
        input_ids = convert_tokens_to_ids(tokenizer, tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # ---------------处理target----------------
        if labels is not None:
            # Notes: label_id中不包括[CLS]和[SEP]
            label_id = [label_map[l] for l in labels]
            label_padding = [-1] * (max_seq_length - len(label_id))
            label_id += label_padding
        else:
            label_id = [-1] * max_seq_length

        # output_mask用来过滤bert输出中sub_word的输出,只保留单词的第一个输出(As recommended by jocob in his paper)
        # 此外，也是为了适应crf
        output_mask = [0] + len(tokens_a) * [1] + [0]
        output_mask += padding

        # ----------------处理后结果-------------------------
        # for example, in the case of max_seq_length=10:
        # raw_data:          春 秋 忽 代 谢le
        # token:       [CLS] 春 秋 忽 代 谢 #le [SEP]
        # input_ids:     101 2  12 13 16 14 15   102   0 0 0
        # input_mask:      1 1  1  1  1  1   1     1   0 0 0
        # label_id:          T  T  O  O  O
        # output_mask:     0 1  1  1  1  1   0     0   0 0 0
        # --------------看结果是否合理------------------------

        # if ex_index < 1:
        #     logger.info("-----------------Example-----------------")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("text_a: %s" % example.text_a)
        #     logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("label: %s " % " ".join([str(x) for x in label_id]))
        #     logger.info("output_mask: %s " % " ".join([str(x) for x in output_mask]))
        # ----------------------------------------------------

        feature = InputFeature(input_ids=input_ids,
                               input_mask=input_mask,
                               segment_ids=segment_ids,
                               label_id=label_id,
                               output_mask=output_mask)
        features.append(feature)

    return features
