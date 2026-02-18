
import numpy as np
from .tensor import tensor


class Linear:

    def __init__(self, in_features, out_features, bias=True):

        # create init tensor
        self.weight = tensor(np.random.randn(in_features, out_features))
        self._is_bias = bias
        if self._is_bias:
            self.bias = tensor(np.random.randn(1, out_features))
        else:
            self.bias = None

    def __call__(self, input: tensor):
        return self.forward(input)

    def forward(self, input: tensor):
        
        if self._is_bias:
            return input @ self.weight + self.bias
        else:
            return input @ self.weight


class Tanh:
    def __call__(self, input: tensor):
        return self.forward(input)

    def forward(self, input: tensor):
        return input.tanh()


class Sigmoid:
    def __call__(self, input: tensor):
        return self.forward(input)

    def forward(self, input: tensor):
        return input.sigmoid()


class ReLU:
    def __call__(self, input: tensor):
        return self.forward(input)

    def forward(self, input: tensor):
        return input.relu()




