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


def delete_similar(predict_true, predict_true_dislike):
    with open(predict_true, 'r') as file_1, open(predict_true_dislike, 'w') as file_2:
        predict = file_1.readlines()
        for index in range(len(predict) // 6):
            line_1 = predict[6 * index]
            line_2 = predict[6 * index + 1]
            line_3 = predict[6 * index + 2]
            line_4 = predict[6 * index + 3]
            line_5 = predict[6 * index + 4]
            line_6 = predict[6 * index + 5]

            if line_4.split(': ')[-1] != line_5.split(': ')[-1]:
                file_2.write(line_1)
                file_2.write(line_2)
                file_2.write(line_3)
                file_2.write(line_4)
                file_2.write(line_5)
                file_2.write(line_6)


def rule_4(predicts):
    for index in range(1, len(predicts)):
        before = predicts[index - 1]
        after = predicts[index]
        if after.startswith('I-') and (before not in [after, 'B' + after[1:]]):
            predicts[index] = 'B' + after[1:]

    return predicts


def rule_5(predicts, x_0_0_length):
    if x_0_0_length == 1 and predicts[0] == 'B-LOC' and predicts[1] == 'B-LOC':
        predicts[0] = 'O'

    return predicts


if __name__ == '__main__':
    predict_dir = '../ner_eda/predict.txt'
    true_dir = '../ner_eda/true.txt'
    predict_true_dir = '../ner_eda/predict_true.txt'
    predict_true_dir_dislike = '../ner_eda/predict_true_dislike.txt'

    # generate(predict_dir, true_dir, predict_true_dir)

    # delete_similar(predict_true_dir, predict_true_dir_dislike)

    print(rule_5(['B-LOC', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'B-LOC', 'I-LOC', 'I-LAW', 'I-LAW', 'I-LAW'], 1))
