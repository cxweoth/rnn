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
            ## First momentum （平均值）
            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            ## Second momentum （變異數）
            v[:] = self.beta2 * v + (1 - self.beta2) * (grad * grad)

            # bias correction
            ## 一開始 m,v 會偏小，所以當 t=1 時除掉較小的數，隨著 t 上升，會除掉更大的數；以此可以讓一開始的 momentum contribute 多一點；避免 m,v 一開始很小導致不太動
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            # update params
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            ## 下面是對於整個 Adam 的說明：
            ## 以更新的公式來看
            ## 更新方向取決於：m_hat
            ## 更新大小取決於：1/(np.sqrt(v_hat))
            ## 當 gradient 絕對值大的時候 -> 更新會變小
            ## 當 gradient 絕對值小的時候 -> 更新會變大
            ## epsilon 是為了避免分母是零，所以加一個小小的值
            #####################################
            ## 更新方向為何要這樣做的說明例子：（避免 gradient 方向變動太快，一下左一下右 (一下正一下負))
            ## 真實方向：→→→→→→→
            ## 但觀察到的 gradient：
            ## → ← → → ← →
            ## momzentum 平均後：
            ## → → → → →
            #####################################
            ## 更新大小為何要有這樣特性的說明例子：
            # ------------------------------------------------------------
            # 山坡直覺圖：
            #
            # 想像 loss 是一個狹窄的山谷：
            #
            #           |
            #           |
            #          / \
            #         /   \
            #        /     \
            #       /       \
            #
            #
            # 如果你站在山坡上：
            #
            #        /
            #       /
            #      /
            #     o
            #
            # 這裡 gradient 很大（坡很陡）
            #
            #
            # ------------------------------------------------------------
            # 如果 step 很大（像 SGD）：
            #
            #        /
            #       /
            #      /
            #     o------------>
            #
            # 會直接衝到另一邊
            #
            #
            # 下一步：
            #
            # <------------o
            #
            # 會來回震盪（oscillation）
            #
            #
            #        /\
            #       /  \
            #      /    \
            #
            # 永遠難以穩定收斂到谷底
            #
            #
            # ------------------------------------------------------------
            # 如果 step 較小（Adam 的效果）：
            #
            #        /
            #       /
            #      /
            #     o-->
            #       -->
            #         -->
            #           o   ← 穩定滑到谷底
            #
            #
            # ------------------------------------------------------------
            # 核心原因：
            #
            # gradient 大
            # = 坡很陡
            # = 區域變化很快
            # = 走太遠會 overshoot
            #
            # 所以需要較小 step 才能穩定收斂
            #
            #
            # Adam 用：
            #
            #     step = m_hat / sqrt(v_hat)
            #
            # 自動在 gradient 大時減小 step
            #
            # 避免震盪與發散
            # ------------------------------------------------------------
            


