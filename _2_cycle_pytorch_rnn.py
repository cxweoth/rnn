import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data_gen.data_reader import read_two_seperate_circle_data


# ---------------------------------------------------------
# 2. 準備資料
# ---------------------------------------------------------
raw_seqs = read_two_seperate_circle_data()
num_sequences = len(raw_seqs)

all_inputs = [torch.from_numpy(s[:-1]).unsqueeze(1) for s in raw_seqs]
all_targets = [torch.from_numpy(s[1:]).unsqueeze(1) for s in raw_seqs]


# ---------------------------------------------------------
# 3. RNN
# ---------------------------------------------------------
class Simple2DRNN(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.rnn = nn.RNN(2, hidden_dim, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, 2)

        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)

    def forward_step(self, x, h):
        out, h = self.rnn(x, h)
        return self.fc(out), h


hidden_dim = 16
model = Simple2DRNN(hidden_dim)

c0_list = nn.ParameterList([
    nn.Parameter(torch.randn(1,1,hidden_dim) * 0.1)
    for _ in range(num_sequences)
])

optimizer = optim.Adam(
    list(model.parameters()) + list(c0_list.parameters()),
    lr=0.003
)

criterion = nn.MSELoss()


# ---------------------------------------------------------
# 4. Free-running rollout training
# ---------------------------------------------------------
def rollout_loss(x_seq, y_seq, h0, teacher_forcing_ratio=0.5):

    T = x_seq.shape[0]

    h = h0

    x = x_seq[0:1]

    loss = 0

    for t in range(T):

        out, h = model.forward_step(x, h)

        loss += criterion(out, y_seq[t:t+1])

        # hybrid input selection
        if np.random.rand() < teacher_forcing_ratio:
            x = x_seq[t:t+1]     # teacher forcing
        else:
            x = out.detach()     # free running

    return loss


print("Training...")
all_epoch = 10000
for epoch in range(all_epoch):

    total_loss = 0

    # gradually reduce teacher forcing
    teacher_forcing_ratio = max(0.8 * (1 - epoch/all_epoch), 0.1)

    for i in range(num_sequences):

        optimizer.zero_grad()

        loss = rollout_loss(
            all_inputs[i],
            all_targets[i],
            c0_list[i],
            teacher_forcing_ratio
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/num_sequences:.6f}")


# ---------------------------------------------------------
# 5. 繪圖（丟掉 transient + PCA hidden）
# ---------------------------------------------------------
def draw_direction_arrow(ax, data, color, interval=30):
    for i in range(0, len(data)-2, interval):
        ax.annotate(
            '',
            xy=data[i+1],
            xytext=data[i],
            arrowprops=dict(arrowstyle='->', color=color, lw=2)
        )


model.eval()
plt.figure(figsize=(14,6))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

colors = ['#7b2cbf', '#2d6a4f']

with torch.no_grad():
    for i in range(num_sequences):

        h = c0_list[i].clone()
        x = all_inputs[i][0:1].clone()

        preds = []
        states = []

        rollout_steps = 1500
        warmup = 500   # 🔥 丟掉前面 transient

        for _ in range(rollout_steps):
            out, h = model.forward_step(x, h)
            preds.append(out.squeeze().numpy())
            states.append(h.squeeze().numpy())
            x = out

        preds = np.array(preds)[warmup:]
        states = np.array(states)[warmup:]

        # -------------------------
        # Output Space
        # -------------------------
        ax1.plot(
            preds[:,0], preds[:,1],
            color=colors[i],
            alpha=0.8,
            label=f"$c_0^{i}$"
        )
        draw_direction_arrow(ax1, preds, colors[i])

        # -------------------------
        # Hidden Space (PCA)
        # -------------------------
        pca = PCA(n_components=2)
        states_2d = pca.fit_transform(states)

        ax2.plot(
            states_2d[:,0], states_2d[:,1],
            color=colors[i],
            alpha=0.8,
            label=f"$c_0^{i}$"
        )
        draw_direction_arrow(ax2, states_2d, colors[i])


# -------------------------
# 美化
# -------------------------
ax1.set_title("Output Space (limit cycle only)")
ax1.set_xlabel("$x^0$")
ax1.set_ylabel("$x^1$")
ax1.axis('equal')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

ax2.set_title("Hidden State Space (PCA)")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()