import numpy as np
from icecream import ic

class Network:
    # sizes = [4,3,2]
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(size, 1) for size in sizes[1:]]
        self.weights = [np.random.randn(row, col) for row, col in zip(sizes[1:], sizes[:-1])]

    # 前馈
    def FeedForward(self, inputs):
        a = inputs
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    # 学习
    def SDG(self, train_data, epoches, eta, mini_batch_len, test_data=None):
        n = len(train_data)
        for i in range(epoches):

            mini_batches = [train_data[i:i+mini_batch_len] for i in range(0, n, mini_batch_len)]
            for mini_batch in mini_batches:
                m = len(mini_batch)
                mb_weights, mb_biases = self.update_mini_batch(mini_batch)
                self.weights = [w - (eta / m) * mw for w, mw in zip(self.weights, mb_weights)]
                self.biases = [b - (eta / m) * mb for b, mb in zip(self.biases, mb_biases)]

            if test_data:
                num_test = len(test_data)
                ic("index :{}, result: {} / {}".format(i, self.evaluate(test_data), num_test))
            else:
                ic(f"{i} has completed")

    # 单批次学习
    def update_mini_batch(self, mini_batch):
        mb_weights = [np.zeros(weight.shape) for weight in self.weights]
        mb_biases = [np.zeros(bias.shape) for bias in self.biases]
        for x, y in mini_batch:
            o_weight, o_bias = self.backForward(x, y)
            mb_weights = [mw + ow for mw, ow in zip(mb_weights, o_weight)]
            mb_biases = [mb + ob for mb, ob in zip(mb_biases, o_bias)]
        return mb_weights, mb_biases

    # 单样本学习
    def backForward(self, x, y):
        o_weight = [np.zeros(weight.shape) for weight in self.weights]
        o_bias = [np.zeros(bias.shape) for bias in self.biases]

        activations = [x]
        zs = []
        a = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1])
        o_bias[-1] = delta
        o_weight[-1] = np.dot(delta, np.transpose(activations[-2]))

        for l in range(2, self.num_layers):
            delta = np.dot(np.transpose(self.weights[-l+1]), delta) * sigmoid_derivative(zs[-l])
            o_bias[-l] = delta
            o_weight[-l] = np.dot(delta, np.transpose(activations[-l-1]))

        return o_weight, o_bias

    # 成本函数求导
    def cost_derivative(self, output, y):
        return output - y

    # 评估
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.FeedForward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))