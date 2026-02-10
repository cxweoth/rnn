import numpy as np

class tensor:
    
    def __init__(self, data, grad_fn=None, requires_grad=True):
        self.data = np.array(data)
        self.grad = None
        self.grad_fn = grad_fn
        self.requires_grad = requires_grad

    def _accumulate_grad(self, g):
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += g

    def __add__(self, other):
        return tensor(self.data + other.data, grad_fn=AddBackward(self, other))

    def __matmul__(self, other):
        return tensor(self.data @ other.data, grad_fn=MatMulBackward(self, other))
    
    def tanh(self):
        res_data = np.tanh(self.data)
        # Create new tensor and link it to TanhBackward
        return tensor(res_data, grad_fn=TanhBackward(self, res_data))
    
    def sigmoid(self):
        res_data = 1.0 / (1.0 + np.exp(-self.data))
        return tensor(res_data, grad_fn=SigmoidBackward(self, res_data))
    
    def relu(self):
        res_data = np.maximum(0.0, self.data)
        return tensor(res_data, grad_fn=ReLUBackward(self, self.data))

    def backward(self, grad_output=None):
        if grad_output is None:
            grad_output = np.ones_like(self.data)

        topo = []
        visited = set()

        def dfs(t):
            tid = id(t)
            if tid in visited:
                return
            visited.add(tid)
            if t.grad_fn is not None:
                for p in t.grad_fn.parents():
                    dfs(p)
            topo.append(t)

        dfs(self)

        grads = {id(self): grad_output}

        for t in reversed(topo):
            g_out = grads.get(id(t))
            if g_out is None:
                continue

            # (簡化版) 這裡對所有 requires_grad tensor 都累積；要更像 PyTorch 可改成只對 leaf 累積
            t._accumulate_grad(g_out)

            if t.grad_fn is None:
                continue

            # IMPORTANT: backward returns a list of (parent, grad_parent), not dict
            for p, gp in t.grad_fn.backward(g_out):
                if not p.requires_grad:
                    continue
                pid = id(p)
                if pid in grads:
                    grads[pid] = grads[pid] + gp
                else:
                    grads[pid] = gp


class AddBackward:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def parents(self):
        return [self.left, self.right]

    def backward(self, grad_output):
        # Use list so duplicated parents (e.g., z+z) don't overwrite
        return [
            (self.left, grad_output),
            (self.right, grad_output),
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
        self.parent_node = parent
        self.out_data = out_data  # Store tanh(x) to reuse in backward

    def parents(self):
        return [self.parent_node]

    def backward(self, grad_output):
        # Derivative of tanh: (1 - tanh(x)^2)
        grad = (1 - self.out_data**2) * grad_output
        return [(self.parent_node, grad)]
    

class SigmoidBackward:
    def __init__(self, parent, out_data):
        self.parent_node = parent
        self.out_data = out_data

    def parents(self):
        return [self.parent_node]

    def backward(self, grad_output):
        # Derivative: sigmoid(x) * (1 - sigmoid(x))
        grad = self.out_data * (1 - self.out_data) * grad_output
        return [(self.parent_node, grad)]


class ReLUBackward:
    def __init__(self, parent, input_data):
        self.parent_node = parent
        self.input_data = input_data

    def parents(self):
        return [self.parent_node]

    def backward(self, grad_output):
        # Derivative: 1 if x > 0 else 0
        grad = grad_output.copy()
        grad[self.input_data <= 0] = 0
        return [(self.parent_node, grad)]

