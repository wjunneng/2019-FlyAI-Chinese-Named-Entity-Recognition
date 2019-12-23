# -*- coding:utf-8 -*-

def generate(predict_dir, true_dir, predict_true_dir):
    with open(predict_dir, 'r') as file_1, open(true_dir, 'r') as file_2, open(predict_true_dir, 'w') as file_3:
        predict = file_1.readlines()
        true = file_2.readlines()
        for index in range(len(true)):
            file_3.write(predict[6 * index])
            file_3.write(predict[6 * index + 1])
            file_3.write(predict[6 * index + 2])
            file_3.write(predict[6 * index + 3])
            file_3.write('trueicts: ' + true[index])
            file_3.write(predict[6 * index + 4])


def rule_2(predicts, a_labels):
    predicts = ' '.join(predicts)
    for a_label in a_labels:
        b_label = "B" + a_label[1:]
        predicts = predicts.replace("O " + a_label, "O " + b_label)

    return predicts.split(' ')


def rule_3(predicts):
    predicts = ' '.join(predicts)
    # b_labels = ['B-LAW', 'B-CRIME', 'B-TIME', 'B-ORG', 'B-PER']
    #          'B-ROLE', 'B-LAW', 'B-LOC', 'B-CRIME', 'B-TIME', 'B-ORG', 'B-PER'
    b_labels = ['B-ORG']
    for b_label in b_labels:
        a_label = "I" + b_label[1:]
        in_end = predicts.split(b_label + ' ' + b_label)
        in_start = in_end[0].strip()
        in_end = in_end[1:]

        # index: / 1:前面有改动 / -1:前面没有改动
        index = -1
        istart = False

        # 前面是以b_label结束的
        if in_start.endswith(b_label):
            index = 1

        for step in in_end:
            step = step.strip()
            # 开始的时候是为空
            if in_start == '':
                in_start = b_label + ' ' + b_label
                index = 1
                istart = True

            # 当前step是为空时
            if step == '':
                if index == 1:
                    # 前面经过改动过
                    in_start = in_start + ' ' + a_label + ' ' + a_label
                else:
                    # 前面没经过改动过
                    in_start = in_start + ' ' + b_label + ' ' + a_label
                    index = 1
            else:
                # 当前是以step开头时
                if step.startswith(b_label):
                    if index == 1:
                        # 前面经过改动过
                        if istart:
                            in_start = in_start + ' I' + step[1:]
                        else:
                            in_start = in_start + ' ' + a_label + ' ' + a_label + ' I' + step[1:]
                    else:
                        # 前面没经过改动过
                        if istart:
                            in_start = in_start + ' ' + step
                        else:
                            in_start = in_start + ' ' + b_label + ' ' + a_label + ' ' + step
                        index = -1
                else:
                    if istart:
                        in_start = in_start + ' ' + step
                    else:
                        in_start = in_start + ' ' + b_label + ' ' + a_label + ' ' + step
                    index = -1

                # 前面是以b_label结束的
                if in_start.endswith(b_label):
                    index = 1
                else:
                    index = -1
            istart = False

        # 去除矛盾的选项
        in_start = in_start.replace(a_label + " " + b_label, a_label + " " + a_label)
        predicts = in_start

    return predicts.split(' ')


if __name__ == '__main__':
    predict_dir = '../ner_eda/predict.txt'
    true_dir = '../ner_eda/true.txt'
    predict_true_dir = '../ner_eda/predict_true.txt'

    generate(predict_dir, true_dir, predict_true_dir)

    # predicts = rule_3(
    #     ['B-ORG', 'B-ORG', 'B-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'B-ORG', 'O', 'O', 'B-PER', 'B-PER', 'O', 'O', 'O',
    #      'B-PER', 'O', 'O', 'O'])
    # print(predicts)

    # predicts = rule_2(
    #     ['I-LAW', 'I-LAW', 'I-LAW', 'I-LAW', 'I-LAW', 'I-LAW', 'I-LAW', 'O', 'I-LAW', 'I-LAW', 'O', 'I-LAW', 'I-LAW',
    #      'B-LAW', 'I-LAW', 'I-LAW'],
    #     a_labels=['I-ROLE', 'I-LAW', 'I-LOC', 'I-CRIME', 'I-TIME', 'I-ORG', 'I-PER'])
    # print(predicts)
