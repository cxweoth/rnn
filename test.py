from rnn.functions import Linear
from rnn.loss import MSELoss
from rnn.optimizer import SGD
from rnn.tensor import tensor

# model
layer = Linear(2, 1)

# collect parameters
params = [layer.weight, layer.bias]

# optimizer
optimizer = SGD(params, lr=0.01)

# data
x = tensor([[1.0, 2.0]])
y = tensor([[0.0]])

# forward
pred = layer(x)

# loss
loss_fn = MSELoss()
loss = loss_fn(pred, y)

# backward
optimizer.zero_grad()
loss.backward()

# update
optimizer.step()

print(layer.weight.data)