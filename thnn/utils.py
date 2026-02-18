import numpy as np
from .tensor import tensor


def xavier_uniform_(W):
    """
    Exact equivalent of PyTorch nn.init.xavier_uniform_
    """
    fan_in, fan_out = W.shape

    limit = np.sqrt(6.0 / (fan_in + fan_out))

    W[:] = np.random.uniform(
        low  = -limit,
        high = +limit,
        size = W.shape
    )

def orthogonal_(tensor_data, gain=1.0):
    """
    Orthogonal initialization (NumPy version)

    tensor_data: numpy array (weight.data)
    gain: scaling factor (default=1.0, tanh 可用 1.0)

    Usage:
        orthogonal_(self.hh.weight.data)
    """

    shape = tensor_data.shape

    if len(shape) < 2:
        raise ValueError("Only tensors with 2+ dimensions are supported")

    rows = shape[0]
    cols = shape[1]

    # flatten to 2D
    flat_shape = (rows, cols)

    # random normal
    a = np.random.randn(*flat_shape).astype(np.float32)

    # QR decomposition
    q, r = np.linalg.qr(a)

    # make Q uniform
    d = np.diag(r)
    ph = np.sign(d)
    q *= ph

    # apply gain
    q = gain * q

    # reshape back
    tensor_data[:] = q.reshape(shape)

    return tensor_data


def zeros_(b):
    """
    Exact equivalent of PyTorch nn.init.zeros_
    """
    b[:] = 0.0

def clip_grad_norm(params, max_norm=1.0, eps=1e-6):
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float((p.grad ** 2).sum())
    total = total ** 0.5
    if total > max_norm:
        scale = max_norm / (total + eps)
        for p in params:
            if p.grad is not None:
                p.grad *= scale
    return total

def rollout_one(model, c0, x0, steps=1000):

    h = tensor(c0.data[0])
    x = tensor(x0.data[0])

    preds = [x.data.squeeze()]
    states = [h.data.squeeze()]

    for _ in range(steps):

        h = model.act(
            model.ih(x) +
            model.hh(h)
        )

        x = model.fc(h)

        preds.append(x.data.squeeze())
        states.append(h.data.squeeze())

    return np.array(preds), np.array(states)


