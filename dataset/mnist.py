import gzip
import struct
from pathlib import Path

from icecream import ic
import numpy as np


current_dir = Path(__file__).parent

def read_images(path):
    with gzip.open(path, 'rb') as f:
        magic, num_images = struct.unpack('>II', f.read(8))
        num_rows, num_cols = struct.unpack('>II', f.read(8))
        num_pixels = num_rows * num_cols
        data = np.frombuffer(f.read(), dtype='uint8').reshape(num_images, num_pixels, 1)
        #归一化处理
        data.astype(np.float32)
        data = data / 255.0
    return data

def read_labels(path):
    with gzip.open(path, 'rb') as f:
        f.read(8)
        data = np.frombuffer(f.read(), dtype='uint8')
    return data

def load_data():
    train_inputs = read_images(current_dir / 'train-images-idx3-ubyte.gz')
    train_outputs = read_labels(current_dir / 'train-labels-idx1-ubyte.gz')
    test_inputs = read_images(current_dir / 't10k-images-idx3-ubyte.gz')
    test_outputs = read_labels(current_dir / 't10k-labels-idx1-ubyte.gz')
    return train_inputs, train_outputs, test_inputs, test_outputs


def load_data_wrapper():
    train_inputs, train_outputs, test_inputs, test_outputs = load_data()
    train_data = zip(train_inputs, [convert_vector(output) for output in train_outputs])
    test_data = zip(test_inputs, test_outputs)
    return list(train_data), list(test_data)

def convert_vector(output):
    vector = np.zeros((10, 1))
    vector[output] = 1.0
    return vector


load_data_wrapper()

