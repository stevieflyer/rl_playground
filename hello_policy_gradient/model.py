import numpy as np
import pickle

from config import resume, H, D
from steve_rl.math import sigmoid


class PolicyGradientPongAgent:
    """
    Policy Gradient Pong Agent

    The model inner network is an MLP with ReLU activation
    and a sigmoid output neuron.

    :param input_dim: input dimensionality
    :param hidden_dim: number of hidden layer neurons
    """
    def __init__(self, input_dim: int, hidden_dim: int, lr: float = 1e-4, decay_rate: float = 0.99):
        self.weights = {'W1': np.random.randn(hidden_dim, input_dim) / np.sqrt(input_dim),
                        'W2': np.random.randn(hidden_dim) / np.sqrt(hidden_dim)}
        """model weights"""
        self.lr = lr
        """learning rate"""
        self.decay_rate = decay_rate
        """decay factor for RMSProp leaky sum of $grad^2$"""
        # update buffers that add up gradients over a batch
        self.__grad_buffer = {k: np.zeros_like(v) for k, v in self.weights.items()}
        """update buffers that add up gradients over a batch, used to simulate the batch gradient descent"""
        # rmsprop memory
        self.__rmsprop_cache = {k: np.zeros_like(v) for k, v in self.weights.items()}
        """rmsprop buffers, used to simulate the batch gradient descent"""

    def forward(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        forward function of policy network

        :param x: input
        :return: probability of taking action 2, and hidden state
        """
        h = np.dot(self.weights['W1'], x)
        h[h < 0] = 0
        logp = np.dot(self.weights['W2'], h)
        p = sigmoid(logp)
        return p, h

    def backward(self, eph, epx, epdlogp):
        """
        Backward pass. (eph is array of intermediate hidden states)
        backward pass will calculate the grad and accumulate it at
        `self.__grad_buffer`

        :param eph: array of intermediate hidden states
        :param epx: array of input states, for pong, it should be 80x80 image
        :param epdlogp: array of gradient of log probability
        :return: None
        """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.weights['W2'])
        dh[eph <= 0] = 0
        dW1 = np.dot(dh.T, epx)
        grad = {'W1': dW1, 'W2': dW2}
        for k in self.weights:
            self.__grad_buffer[k] += grad[k]  # accumulate grad over batch

    def update_weights(self):
        """
        update weights using rmsprop algorithm
        :return: None
        """
        for k, v in self.weights.items():
            g = self.__grad_buffer[k]
            self.__rmsprop_cache[k] = self.decay_rate * self.__rmsprop_cache[k] + (1 - self.decay_rate) * g ** 2
            self.weights[k] += self.lr * g / (np.sqrt(self.__rmsprop_cache[k]) + 1e-5)
            self.__grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

    def load_model_weights(self, path):
        """Load mode weight from a file"""
        self.weights = pickle.load(open(path, 'rb'))

    def save_model_weights(self, path):
        """Save mode weight to a file"""
        pickle.dump(self.weights, open(path, 'wb'))
