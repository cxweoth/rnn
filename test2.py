import numpy as np
from rnn.tensor import tensor
from rnn.functions import Linear, Tanh


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        
        # 使用封裝好的 Linear 層
        self.i2h = Linear(input_size, hidden_size)  # 處理 x_t
        self.h2h = Linear(hidden_size, hidden_size)  # 處理 h_prev
        self.h2o = Linear(hidden_size, output_size)  # 處理最終 output
        
        # 使用封裝好的 Tanh 層
        self.activation = Tanh()

    def __call__(self, x, h_prev=None):
        return self.forward(x, h_prev)

    def forward(self, x, h_prev=None):
        """
        x: 預期為一個 list of tensor，長度為 seq_len
           每個 tensor 形狀為 (batch_size, input_size)
        """
        if x and len(x) > 0:
            batch_size = x[0].data.shape[0]
        else:
            raise ValueError("Input sequence x cannot be empty")

        # 初始化隱藏狀態
        if h_prev is None:
            h_prev = tensor(np.zeros((batch_size, self.hidden_size)))
        
        hidden_history = []
        h_t = h_prev
        
        # 遍歷時間步
        for x_t in x:
            # RNN 核心公式：h_t = Tanh( Linear_i2h(x_t) + Linear_h2h(h_t) )
            # 這裡完全使用你定義好的 __call__ 介面
            
            i_part = self.i2h(x_t)      # 得到 tensor
            h_part = self.h2h(h_t)      # 得到 tensor
            
            # 透過 tensor 的 __add__ 合併，再丟進 Tanh 層
            h_t = self.activation(i_part + h_part)
            
            hidden_history.append(h_t)
            
        # 將最後一步的隱藏狀態轉換為輸出
        output = self.h2o(h_t)
        
        return output, hidden_history