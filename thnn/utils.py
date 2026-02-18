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


def zeros_(b):
    """
    Exact equivalent of PyTorch nn.init.zeros_
    """
    b[:] = 0.0


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


