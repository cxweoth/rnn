# rnn.py
import numpy as np
from .tensor import tensor, stack
from .functions import Linear, Tanh
from .utils import xavier_uniform_, zeros_, orthogonal_


class RNN_2D:
    def __init__(self):
        self.ih = Linear(2, 2, bias=True)
        self.hh = Linear(2, 2, bias=True)
        self.fc = Linear(2, 2, bias=True)
        self.act = Tanh()

        # weight initialization
        xavier_uniform_(self.ih.weight.data)
        xavier_uniform_(self.hh.weight.data)
        xavier_uniform_(self.fc.weight.data)

        # bias initialization
        zeros_(self.ih.bias.data)
        zeros_(self.hh.bias.data)
        zeros_(self.fc.bias.data)

    def forward(self, x: tensor, h0: tensor, free_run: bool = False):
        """
        free_run=False → teacher forcing
        free_run=True  → free running
        """
        T, B, D = x.data.shape

        # h0 is expected to be shape (1, B, 2) or (B, 2)
        h = h0[0] if len(h0.data.shape) == 3 else h0  # (B, 2)

        outputs = []

        # initial input: (B, 2)
        x_t = x[0]

        for t in range(T):
            # update hidden
            h = self.act(self.ih(x_t) + self.hh(h))

            # compute output
            y = self.fc(h)
            outputs.append(y)

            # choose next input
            if free_run:
                # IMPORTANT: keep graph (do NOT use y.data)
                x_t = y
            else:
                if t + 1 < T:
                    x_t = x[t + 1]

        output = stack(outputs)      # (T, B, 2)
        h_final = stack([h])         # (1, B, 2)

        return output, h_final

    def __call__(self, x, h0, free_run=False):
        return self.forward(x, h0, free_run)

    def parameters(self):
        params = []
        for layer in [self.ih, self.hh, self.fc]:
            params.append(layer.weight)
            if layer.bias is not None:
                params.append(layer.bias)
        return params

class RNN_2D_Customized_Hidden_Space:
    def __init__(self, hidden_dim=16):
        self.hidden_dim = hidden_dim
        self.ih = Linear(2, hidden_dim, bias=True)
        self.hh = Linear(hidden_dim, hidden_dim, bias=True)
        self.fc = Linear(hidden_dim, 2, bias=True)
        self.act = Tanh()

        xavier_uniform_(self.ih.weight.data)
        orthogonal_(self.hh.weight.data)
        xavier_uniform_(self.fc.weight.data)

        zeros_(self.ih.bias.data)
        zeros_(self.hh.bias.data)
        zeros_(self.fc.bias.data)

    def forward(self, x: tensor, h0: tensor, free_run: bool = False):
        T, B, D = x.data.shape

        # h0: (1,B,H) or (B,H)
        if len(h0.data.shape) == 3:
            assert h0.data.shape[0] == 1
            h = h0[0]
        else:
            h = h0

        outputs = []
        x_t = x[0]

        for t in range(T):
            h = self.act(self.ih(x_t) + self.hh(h))
            y = self.fc(h)
            outputs.append(y)

            if free_run:
                x_t = y
            else:
                if t + 1 < T:
                    x_t = x[t + 1]

        output = stack(outputs)   # (T, B, 2)
        h_final = stack([h])      # (1, B, hidden_dim)
        return output, h_final

    def __call__(self, x, h0, free_run=False):
        return self.forward(x, h0, free_run)

    def parameters(self):
        params = []
        for layer in [self.ih, self.hh, self.fc]:
            params.append(layer.weight)
            if layer.bias is not None:
                params.append(layer.bias)
        return params

    def state_dict(self):
        state = {
            "ih.weight": self.ih.weight.data.tolist(),
            "hh.weight": self.hh.weight.data.tolist(),
            "fc.weight": self.fc.weight.data.tolist(),
        }
        if self.ih.bias is not None:
            state["ih.bias"] = self.ih.bias.data.tolist()
        if self.hh.bias is not None:
            state["hh.bias"] = self.hh.bias.data.tolist()
        if self.fc.bias is not None:
            state["fc.bias"] = self.fc.bias.data.tolist()
        return state

    def load_state_dict(self, state):
        self.ih.weight.data[:] = np.asarray(state["ih.weight"], dtype=self.ih.weight.data.dtype)
        self.hh.weight.data[:] = np.asarray(state["hh.weight"], dtype=self.hh.weight.data.dtype)
        self.fc.weight.data[:] = np.asarray(state["fc.weight"], dtype=self.fc.weight.data.dtype)

        if "ih.bias" in state and self.ih.bias is not None:
            self.ih.bias.data[:] = np.asarray(state["ih.bias"], dtype=self.ih.bias.data.dtype)
        if "hh.bias" in state and self.hh.bias is not None:
            self.hh.bias.data[:] = np.asarray(state["hh.bias"], dtype=self.hh.bias.data.dtype)
        if "fc.bias" in state and self.fc.bias is not None:
            self.fc.bias.data[:] = np.asarray(state["fc.bias"], dtype=self.fc.bias.data.dtype)