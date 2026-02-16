import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from rnn.functions import Linear, Tanh
from data_gen_by_gesture.data_reader import read_one_circle_data


# ---------------------------------------------------------
# 1. generate 3 seqences of 2D data
# ---------------------------------------------------------
def generate_data(offsets):

    base_raw = read_one_circle_data()

    base = np.array([[p["x"], p["y"]] for p in base_raw],
                    dtype=np.float32)

    sequences = []
    for dx, dy in offsets:
        shifted = base + np.array([[dx, dy]], dtype=np.float32)
        sequences.append(shifted)

    return sequences


# ---------------------------------------------------------
# 2. 準備資料
# ---------------------------------------------------------
num_sequences = 3
offsets = [(0.01, 0.0), (0.0, 0.0), (-0.01, 0.01)]
raw_seqs = generate_data(offsets)

all_inputs = [s[:-1][:, np.newaxis] for s in raw_seqs]
all_targets = [s[1:][:, np.newaxis] for s in raw_seqs]

# ---------------------------------------------------------
# 3. 2D RNN
# ---------------------------------------------------------
class Simple2DRNN:

    def __init__(self):

        # input → hidden
        self.ih = Linear(2, 2)

        # hidden → hidden
        self.hh = Linear(2, 2, bias=False)

        # activation
        self.act = Tanh()

        # output layer
        self.fc = Linear(2, 2)

    def __call__(self, x, h0):
        return self.forward(x, h0)

    def forward(self, x, h0):
        """
        x: list or iterable of tensor, each shape (B, 2)
           或 shape (T, B, 2) 也可以自己拆
        h0: tensor shape (B, 2)
        """

        h = h0
        outputs = []

        # 如果 x 是 (T, B, 2)，先拆成時間步
        if hasattr(x, "shape"):
            T = x.shape[0]
            x_seq = [x[t] for t in range(T)]
        else:
            x_seq = x

        for xt in x_seq:
            h = self.act(self.ih(xt) + self.hh(h))
            y = self.fc(h)
            outputs.append(y)

        return outputs, h


model = Simple2DRNN()

c0_list = nn.ParameterList([
    nn.Parameter(torch.randn(1,1,2) * 0.1)
    for _ in range(num_sequences)
])

optimizer = optim.Adam(
    list(model.parameters()) + list(c0_list.parameters()),
    lr=0.005
)

criterion = nn.MSELoss()


# ---------------------------------------------------------
# 4. 訓練
# ---------------------------------------------------------
print("Training 2D RNN...")

for epoch in range(1000):
    total_loss = 0

    for i in range(num_sequences):
        optimizer.zero_grad()
        output, _ = model(all_inputs[i], c0_list[i])
        loss = criterion(output, all_targets[i])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch+1) % 200 == 0:
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/num_sequences:.6f}")


# ---------------------------------------------------------
# 5. 繪圖（從真正 c0 開始，無 warmup）
# ---------------------------------------------------------

def draw_direction_arrow(ax, data, color, interval=30):
    for i in range(0, len(data)-2, interval):
        p1 = data[i]
        p2 = data[i+1]
        ax.annotate(
            '',
            xy=(p2[0], p2[1]),
            xytext=(p1[0], p1[1]),
            arrowprops=dict(
                arrowstyle='->',
                color=color,
                lw=2,
                mutation_scale=12
            )
        )


model.eval()
plt.figure(figsize=(14,6))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

colors = ['#7b2cbf', '#3a0ca3', '#2d6a4f']

with torch.no_grad():
    for i in range(num_sequences):

        # -------------------------
        # 初始 hidden 與 input
        # -------------------------
        h = c0_list[i].clone()
        x = all_inputs[i][0:1].clone()

        true_c0 = h.detach().cpu().numpy().squeeze()
        true_x0 = x.detach().cpu().numpy().squeeze()

        preds = [true_x0.copy()]
        states = [true_c0.copy()]

        # -------------------------
        # rollout dynamics
        # -------------------------
        for _ in range(1000):
            out, h = model(x, h)
            preds.append(out.detach().cpu().numpy().squeeze())
            states.append(h.detach().cpu().numpy().squeeze())
            x = out

        preds = np.array(preds)
        states = np.array(states)

        # -------------------------
        # Output Space
        # -------------------------
        ax1.plot(
            preds[:,0], preds[:,1],
            color=colors[i],
            alpha=0.7,
            label=f"$c_0^{i}$"
        )
        draw_direction_arrow(ax1, preds, colors[i])

        # -------------------------
        # Hidden Space
        # -------------------------
        ax2.plot(
            states[:,0], states[:,1],
            color=colors[i],
            alpha=0.6,
            label=f"$c_0^{i}$"
        )
        draw_direction_arrow(ax2, states, colors[i])


# -------------------------
# 美化
# -------------------------
ax1.set_title("Output Space ($x^1$ vs $x^0$)")
ax1.set_xlabel("$x^0$")
ax1.set_ylabel("$x^1$")
ax1.axis('equal')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

ax2.set_title("Internal State Space ($c^1$ vs $c^0$)")
ax2.set_xlabel("$c^0$")
ax2.set_ylabel("$c^1$")
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()