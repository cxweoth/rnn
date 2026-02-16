import numpy as np

from rnn.tensor import tensor
from rnn.functions import Linear, Tanh
from rnn.loss import MSELoss
from rnn.optimizer import SGD


# ============================
# Your RNN class
# ============================

class Simple2DRNN:

    def __init__(self):

        self.ih = Linear(2, 2)
        self.hh = Linear(2, 2, bias=False)
        self.act = Tanh()
        self.fc = Linear(2, 2)

    def __call__(self, x, h0):
        return self.forward(x, h0)

    def forward(self, x_seq, h0):

        h = h0
        outputs = []

        for xt in x_seq:

            h = self.act(
                self.ih(xt) + self.hh(h)
            )

            y = self.fc(h)

            outputs.append(y)

        return outputs, h

    # IMPORTANT: parameters for optimizer
    def parameters(self):

        params = []

        for layer in [self.ih, self.hh, self.fc]:

            params.append(layer.weight)

            if layer.bias is not None:
                params.append(layer.bias)

        return params


# ============================
# Create toy dataset
# learn identity sequence
# ============================

def create_sequence():

    seq = []

    seq.append(tensor([[1.0, 0.0]]))
    seq.append(tensor([[0.0, 1.0]]))
    seq.append(tensor([[1.0, 1.0]]))

    return seq


# ============================
# Test function
# ============================

def test_rnn():

    np.random.seed(0)

    rnn = Simple2DRNN()

    optimizer = SGD(
        rnn.parameters(),
        lr=0.05
    )

    loss_fn = MSELoss()

    h0 = tensor(np.zeros((1,2)))

    x_seq = create_sequence()

    target = tensor([[0.5, -0.5]])

    print("\nInitial parameters:")

    print(rnn.fc.weight.data)

    print("\nTraining start:\n")

    for epoch in range(50):

        outputs, hT = rnn(x_seq, h0)

        pred = outputs[-1]

        loss = loss_fn(pred, target)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        print(f"epoch {epoch:02d} loss = {loss.data:.6f}")

    print("\nFinal parameters:")

    print(rnn.fc.weight.data)

    print("\nFinal prediction:")

    outputs, _ = rnn(x_seq, h0)

    print(outputs[-1].data)


# ============================

if __name__ == "__main__":

    test_rnn()