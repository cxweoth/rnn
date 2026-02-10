import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. 資料生成：仿照老師圖中的多樣軌跡
# ---------------------------------------------------------
def generate_teacher_like_data(n_points=120):
    sequences = []
    # 定義三組不同的參數 (半徑x, 半徑y, 中心x, 中心y)
    params = [
        (0.8, 0.9, 0.1, 0.0),   # 序列 0: 稍微扁平且往右偏
        (1.0, 1.0, 0.0, 0.0),   # 序列 1: 標準單位圓
        (1.2, 1.1, -0.1, 0.1)   # 序列 2: 大圓且中心往左上偏
    ]
    
    for rx, ry, ox, oy in params:
        # 隨機起始相位，讓 $c_0$ 學習不同的進入點
        start_phase = np.random.uniform(0, 2 * np.pi)
        theta = np.linspace(start_phase, start_phase + 2 * np.pi, n_points)
        x = rx * np.cos(theta) + ox
        y = ry * np.sin(theta) + oy
        sequences.append(np.column_stack((x, y)).astype(np.float32))
    return sequences

# 準備訓練數據
num_sequences = 3
raw_seqs = generate_teacher_like_data(120)
all_inputs = [torch.from_numpy(s[:-1]).unsqueeze(1) for s in raw_seqs]
all_targets = [torch.from_numpy(s[1:]).unsqueeze(1) for s in raw_seqs]

# ---------------------------------------------------------
# 2. 模型定義：Multi-Context RNN
# ---------------------------------------------------------
class MultiContextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiContextRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, output_size)
        # 初始化權重以維持長時序穩定性
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0)

    def forward(self, x, h0):
        out, h_final = self.rnn(x, h0)
        return self.fc(out), h_final

# ---------------------------------------------------------
# 3. 訓練階段
# ---------------------------------------------------------
hidden_dim = 64
model = MultiContextRNN(2, hidden_dim, 2)
# 為每個序列定義一個獨立且可學習的 c0
c0_list = nn.ParameterList([nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.1) for _ in range(num_sequences)])

optimizer = optim.Adam(list(model.parameters()) + list(c0_list.parameters()), lr=0.005)
criterion = nn.MSELoss()

print("Training for limit cycles...")
for epoch in range(1500):
    total_loss = 0
    for i in range(num_sequences):
        optimizer.zero_grad()
        output, _ = model(all_inputs[i], c0_list[i])
        loss = criterion(output, all_targets[i])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/num_sequences:.6f}")

# ---------------------------------------------------------
# 4. 繪圖與方向標示
# ---------------------------------------------------------
def draw_direction_arrow(ax, data, color, interval=40):
    """在軌跡上每隔一段距離畫一個箭頭標示方向"""
    for i in range(0, len(data) - 5, interval):
        p1 = data[i]
        p2 = data[i+1] # 用緊鄰的點計算切線向量
        ax.annotate('', xy=(p2[0], p2[1]), xytext=(p1[0], p1[1]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2, mutation_scale=15))

model.eval()
plt.figure(figsize=(14, 6))
ax1 = plt.subplot(1, 2, 1) # Output Space
ax2 = plt.subplot(1, 2, 2) # Context Space
colors = ['#7b2cbf', '#3a0ca3', '#2d6a4f'] # 使用類似老師圖中的配色

with torch.no_grad():
    for i in range(num_sequences):
        curr_h = c0_list[i]
        curr_in = all_inputs[i][0:1]
        
        preds, states = [], []
        
        # --- Warm-up 20 steps ---
        for _ in range(20):
            out, curr_h = model.rnn(curr_in, curr_h)
            curr_in = model.fc(out)
            
        # --- Long-term Generation (500 steps) ---
        for _ in range(500):
            out, curr_h = model.rnn(curr_in, curr_h)
            pos = model.fc(out)
            preds.append(pos.squeeze().numpy())
            states.append(curr_h.squeeze().numpy())
            curr_in = pos
            
        preds, states = np.array(preds), np.array(states)
        
        # 繪製 Output Space (x1 vs x0)
        ax1.plot(preds[:, 0], preds[:, 1], color=colors[i], alpha=0.6, label=f'Target-Seq {i}')
        draw_direction_arrow(ax1, preds, colors[i])
        
        # 繪製 Context Space (c1 vs c0) - 取隱藏層前兩個維度
        ax2.plot(states[:, 0], states[:, 1], color=colors[i], alpha=0.5)
        draw_direction_arrow(ax2, states[:, :2], colors[i])
        ax2.scatter(states[0, 0], states[0, 1], color=colors[i], s=50, edgecolors='white', label=f'Initial $c_0$ {i}')

# 圖表美化
ax1.set_title("Output Space ($x^1$ vs $x^0$)")
ax1.set_xlabel("$x^0$")
ax1.set_ylabel("$x^1$")
ax1.axis('equal')
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend()

ax2.set_title("Internal State Space ($c^1$ vs $c^0$)")
ax2.set_xlabel("$c^0$")
ax2.set_ylabel("$c^1$")
ax2.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()