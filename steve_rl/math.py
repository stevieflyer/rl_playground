import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalize(x):
    return (x - np.mean(x)) / np.std(x)