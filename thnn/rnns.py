import numpy as np
from .tensor import tensor, stack
from .functions import Linear, Sigmoid
from .utils import xavier_uniform_, zeros_


class RNN_2D:

    def __init__(self):
        self.ih = Linear(2, 2, bias=True)
        self.hh = Linear(2, 2, bias=True)
        self.fc = Linear(2, 2, bias=True)
        self.act = Sigmoid()

        # weight initialization
        xavier_uniform_(self.ih.weight.data)
        xavier_uniform_(self.hh.weight.data)
        xavier_uniform_(self.fc.weight.data)

        # bias initialization
        zeros_(self.ih.bias.data)
        zeros_(self.hh.bias.data)
        zeros_(self.fc.bias.data)

    def forward(self, x, h0, free_run=False):

        """
        free_run=False → teacher forcing
        free_run=True  → free running
        """

        T, B, D = x.data.shape

        h = tensor(h0.data[0])   # (1,2)

        outputs = []

        # initial input
        x_t = tensor(x.data[0])

        for t in range(T):

            # update hidden
            h = self.act(
                self.ih(x_t) +
                self.hh(h)
            )

            # compute output
            y = self.fc(h)

            outputs.append(y)

            # choose next input
            if free_run:
                x_t = tensor(y.data)   # use model output
            else:
                if t+1 < T:
                    x_t = tensor(x.data[t+1])  # use teacher

        output = stack(outputs)

        h_final = tensor(h.data[np.newaxis,:,:])

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