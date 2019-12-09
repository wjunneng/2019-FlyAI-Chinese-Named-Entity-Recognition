from preprocessing.data_processor import produce_data
from preprocessing.data_loader import create_batch_iter


def start():
    produce_data()
    train_iter, num_train_steps = create_batch_iter("train")


if __name__ == '__main__':
    start()
