import numpy as np

class Network:
    # sizes = [4,3,2]
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(size, 1) for size in sizes[1:]]
        self.weights = [np.random.randn(row, col) for row, col in zip(sizes[1:], sizes[:-1])]

    # 前馈
    def FeedForward(self, inputs):
        a = inputs
        for index in range(self.num_layers - 1):
            a = sigmoid(np.dot(self.weights[index], a) + self.biases[index])
        return a

    # 学习
    def SDG(self, train_data, epoches, eta, mini_batch_len, test_data=None):

        pass

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))