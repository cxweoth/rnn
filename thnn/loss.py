from .tensor import tensor

class MSELoss:

    def __call__(self, prediction: tensor, target: tensor):
        return self.forward(prediction, target)

    def forward(self, prediction: tensor, target: tensor):
        diff = prediction - target
        sq = diff * diff
        return sq.mean()
    
