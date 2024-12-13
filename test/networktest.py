from dataset import mnist
from learn import network

train_data, test_data =  mnist.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SDG(train_data, 30, 3.0, 10, test_data)

