import numpy as np


# =====================================================
# HELPER: reverse numpy broadcasting for gradients
# =====================================================

def _unbroadcast(grad, shape):

    # Remove extra leading dimensions
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    # Sum axes where original shape was 1
    for axis, (gdim, sdim) in enumerate(zip(grad.shape, shape)):
        if sdim == 1 and gdim != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad


# =====================================================
# TENSOR
# =====================================================

class tensor:

    def __init__(self, data, grad_fn=None, requires_grad=True):

        self.data = np.array(data, dtype=np.float32)

        self.grad = None

        self.grad_fn = grad_fn

        self.requires_grad = requires_grad


    # =====================================================
    # accumulate grad
    # =====================================================

    def _accumulate_grad(self, g):

        if not self.requires_grad:
            return

        if self.grad is None:
            self.grad = np.zeros_like(self.data)

        self.grad += g


    # =====================================================
    # BASIC OPERATIONS
    # =====================================================

    def __add__(self, other):

        return tensor(
            self.data + other.data,
            grad_fn=AddBackward(self, other),
            requires_grad=self.requires_grad or other.requires_grad
        )


    def __sub__(self, other):

        return tensor(
            self.data - other.data,
            grad_fn=SubBackward(self, other),
            requires_grad=self.requires_grad or other.requires_grad
        )


    def __mul__(self, other):

        return tensor(
            self.data * other.data,
            grad_fn=MulBackward(self, other),
            requires_grad=self.requires_grad or other.requires_grad
        )


    def __matmul__(self, other):

        return tensor(
            self.data @ other.data,
            grad_fn=MatMulBackward(self, other),
            requires_grad=self.requires_grad or other.requires_grad
        )


    # =====================================================
    # ACTIVATIONS
    # =====================================================

    def tanh(self):

        out = np.tanh(self.data)

        return tensor(
            out,
            grad_fn=TanhBackward(self, out),
            requires_grad=self.requires_grad
        )


    def sigmoid(self):

        out = 1.0 / (1.0 + np.exp(-self.data))

        return tensor(
            out,
            grad_fn=SigmoidBackward(self, out),
            requires_grad=self.requires_grad
        )


    def relu(self):

        out = np.maximum(0.0, self.data)

        return tensor(
            out,
            grad_fn=ReLUBackward(self, self.data),
            requires_grad=self.requires_grad
        )


    # =====================================================
    # REDUCTIONS
    # =====================================================

    def mean(self):

        out = self.data.mean()

        return tensor(
            out,
            grad_fn=MeanBackward(self, self.data.shape),
            requires_grad=self.requires_grad
        )


    # =====================================================
    # BACKWARD
    # =====================================================

    def backward(self, grad_output=None):

        if not self.requires_grad:
            return

        if grad_output is None:
            grad_output = np.ones_like(self.data, dtype=np.float32)

        topo = []

        visited = set()

        def dfs(t):

            if id(t) in visited:
                return

            visited.add(id(t))

            if t.grad_fn is not None:
                for parent in t.grad_fn.parents():
                    dfs(parent)

            topo.append(t)

        dfs(self)

        grads = {id(self): grad_output}

        for t in reversed(topo):

            g_out = grads.get(id(t))

            if g_out is None:
                continue

            t._accumulate_grad(g_out)

            if t.grad_fn is None:
                continue

            for parent, grad_parent in t.grad_fn.backward(g_out):

                if not parent.requires_grad:
                    continue

                pid = id(parent)

                if pid in grads:
                    grads[pid] += grad_parent
                else:
                    grads[pid] = grad_parent


# =====================================================
# STACK
# =====================================================

def stack(tensor_list):

    data = np.stack([t.data for t in tensor_list], axis=0)

    return tensor(
        data,
        grad_fn=StackBackward(tensor_list),
        requires_grad=any(t.requires_grad for t in tensor_list)
    )


# =====================================================
# BACKWARD CLASSES
# =====================================================

class AddBackward:

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def parents(self):
        return [self.left, self.right]

    def backward(self, grad_output):

        grad_left = _unbroadcast(grad_output, self.left.data.shape)

        grad_right = _unbroadcast(grad_output, self.right.data.shape)

        return [
            (self.left, grad_left),
            (self.right, grad_right),
        ]


class SubBackward:

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def parents(self):
        return [self.left, self.right]

    def backward(self, grad_output):

        grad_left = _unbroadcast(grad_output, self.left.data.shape)

        grad_right = _unbroadcast(-grad_output, self.right.data.shape)

        return [
            (self.left, grad_left),
            (self.right, grad_right),
        ]


class MulBackward:

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def parents(self):
        return [self.left, self.right]

    def backward(self, grad_output):

        grad_left = grad_output * self.right.data
        grad_right = grad_output * self.left.data

        grad_left = _unbroadcast(grad_left, self.left.data.shape)
        grad_right = _unbroadcast(grad_right, self.right.data.shape)

        return [
            (self.left, grad_left),
            (self.right, grad_right),
        ]


class MatMulBackward:

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def parents(self):
        return [self.left, self.right]

    def backward(self, grad_output):

        grad_left = grad_output @ self.right.data.T

        grad_right = self.left.data.T @ grad_output

        return [
            (self.left, grad_left),
            (self.right, grad_right),
        ]


class TanhBackward:

    def __init__(self, parent, out_data):
        self.parent = parent
        self.out_data = out_data

    def parents(self):
        return [self.parent]

    def backward(self, grad_output):

        grad = (1.0 - self.out_data**2) * grad_output

        return [(self.parent, grad)]


class SigmoidBackward:

    def __init__(self, parent, out_data):
        self.parent = parent
        self.out_data = out_data

    def parents(self):
        return [self.parent]

    def backward(self, grad_output):

        grad = self.out_data * (1 - self.out_data) * grad_output

        return [(self.parent, grad)]


class ReLUBackward:

    def __init__(self, parent, input_data):
        self.parent = parent
        self.input_data = input_data

    def parents(self):
        return [self.parent]

    def backward(self, grad_output):

        grad = grad_output.copy()

        grad[self.input_data <= 0] = 0

        return [(self.parent, grad)]


class MeanBackward:

    def __init__(self, parent, original_shape):
        self.parent = parent
        self.original_shape = original_shape

    def parents(self):
        return [self.parent]

    def backward(self, grad_output):

        size = np.prod(self.original_shape)

        grad = np.ones(self.original_shape, dtype=np.float32) * (grad_output / size)

        return [(self.parent, grad)]


class StackBackward:

    def __init__(self, parents):
        self._parents = parents

    def parents(self):
        return self._parents

    def backward(self, grad_output):

        grads = []

        for i, parent in enumerate(self._parents):

            grad_piece = grad_output[i]

            grads.append((parent, grad_piece))

        return grads