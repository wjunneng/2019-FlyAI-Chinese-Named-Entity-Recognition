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


if __name__ == '__main__':
    predict_dir = '../ner_eda/predict.txt'
    true_dir = '../ner_eda/true.txt'
    predict_true_dir = '../ner_eda/predict_true.txt'

    generate(predict_dir, true_dir, predict_true_dir)
