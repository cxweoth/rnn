import numpy as np


class Optimizer:
    """
    Base optimizer class
    """

    def __init__(self, params):
        """
        params: list of tensor objects
        """
        self.params = list(params)

    def zero_grad(self):
        """
        Reset gradients (PyTorch-compatible)
        """
        for p in self.params:
            p.grad = None

    def step(self):
        """
        Override in subclass
        """
        raise NotImplementedError
    

class SGD(Optimizer):

    def __init__(self, params, lr=0.01):
        super().__init__(params)
        self.lr = lr

    def step(self):

        for p in self.params:

            if p.grad is None:
                continue

            # gradient descent update
            p.data -= self.lr * p.grad


class SGD_Momentum(Optimizer):

    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params)

        self.lr = lr
        self.momentum = momentum

        self.velocity = {}

        for p in self.params:
            self.velocity[id(p)] = np.zeros_like(p.data)

    def step(self):

        for p in self.params:

            if p.grad is None:
                continue

            v = self.velocity[id(p)]

            v[:] = self.momentum * v - self.lr * p.grad

            p.data += v


class Adam(Optimizer):

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):

        super().__init__(params)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = {}
        self.v = {}
        self.t = 0

        for p in self.params:
            self.m[id(p)] = np.zeros_like(p.data)
            self.v[id(p)] = np.zeros_like(p.data)

    def step(self):

        self.t += 1

        for p in self.params:

            if p.grad is None:
                continue

            m = self.m[id(p)]
            v = self.v[id(p)]

            grad = p.grad

            # update biased moments
            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            v[:] = self.beta2 * v + (1 - self.beta2) * (grad * grad)

            # bias correction
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            # update params
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


