import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from data_gen.data_reader import read_eight_shape_data


# =========================================================
# 1) Data
# =========================================================
def generate_data():
    base_raw = read_eight_shape_data()
    base = np.array([[p["x"], p["y"]] for p in base_raw], dtype=np.float32)  # (T,2)
    return [base]  # num_sequences = 1


raw_seqs = generate_data()
num_sequences = len(raw_seqs)

# Convert to torch: input is x_t, target is x_{t+1}
all_inputs = [torch.from_numpy(s[:-1]).unsqueeze(1) for s in raw_seqs]   # (T-1,1,2)
all_targets = [torch.from_numpy(s[1:]).unsqueeze(1) for s in raw_seqs]   # (T-1,1,2)


# =========================================================
# 2) Model: higher-dim hidden, 2D output
# =========================================================
class Simple2DRNN(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_size=2, hidden_size=hidden_dim, nonlinearity="tanh")
        self.fc = nn.Linear(hidden_dim, 2)

        # init
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        if self.rnn.bias_ih_l0 is not None:
            nn.init.zeros_(self.rnn.bias_ih_l0)
        if self.rnn.bias_hh_l0 is not None:
            nn.init.zeros_(self.rnn.bias_hh_l0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward_step(self, x_t, h_t):
        """
        x_t: (1,1,2)
        h_t: (1,1,H)
        returns:
          y_t: (1,1,2) predicted x_{t+1}
          h_{t+1}: (1,1,H)
        """
        out, h_next = self.rnn(x_t, h_t)     # out: (1,1,H)
        y = self.fc(out)                    # (1,1,2)
        return y, h_next

    def forward_teacher_forcing(self, x_seq, h0):
        """
        x_seq: (T,1,2)
        h0: (1,1,H)
        returns y_seq: (T,1,2)
        """
        out, h_final = self.rnn(x_seq, h0)
        y = self.fc(out)
        return y, h_final


# =========================================================
# 3) Training: rollout loss (free-running), optional scheduled sampling
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_dim = 16
model = Simple2DRNN(hidden_dim=hidden_dim).to(device)

# one sequence -> one trainable initial hidden state
c0_list = nn.ParameterList([
    nn.Parameter(torch.randn(1, 1, hidden_dim, device=device) * 0.1)
    for _ in range(num_sequences)
])

optimizer = optim.Adam(list(model.parameters()) + list(c0_list.parameters()), lr=0.003)
criterion = nn.MSELoss()

# scheduled sampling: start with some teacher forcing then gradually reduce
use_scheduled_sampling = True
teacher_forcing_start = 0.5   # probability at epoch 0
teacher_forcing_end = 0.0     # probability at final epoch

epochs = 5000
print_every = 250


def teacher_forcing_prob(epoch):
    if not use_scheduled_sampling:
        return 0.0
    # linear decay
    p = teacher_forcing_start + (teacher_forcing_end - teacher_forcing_start) * (epoch / max(1, epochs - 1))
    return float(np.clip(p, 0.0, 1.0))


def rollout_loss(x_seq, y_true_seq, h0, p_tf=0.0):
    """
    x_seq: (T,1,2)  (these are ground-truth x_t, t=0..T-1)
    y_true_seq: (T,1,2) (ground-truth x_{t+1}, t=0..T-1)
    Train in auto-regressive mode:
      x_in starts as x_0, then x_{t+1} is either predicted (free run) or gt (teacher forcing) based on p_tf.
    """
    T = x_seq.shape[0]
    h = h0
    x_in = x_seq[0:1]  # (1,1,2) start from ground-truth x0
    loss = 0.0

    for t in range(T):
        y_pred, h = model.forward_step(x_in, h)  # predicts x_{t+1}
        loss = loss + criterion(y_pred, y_true_seq[t:t+1])

        # choose next input
        if p_tf > 0.0:
            # scheduled sampling: sometimes feed ground truth, sometimes feed model output
            if torch.rand(()) < p_tf:
                x_in = x_seq[t:t+1] if t < T - 1 else y_pred.detach()
            else:
                x_in = y_pred.detach()
        else:
            x_in = y_pred.detach()

    return loss


print("Training (rollout loss)...")
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    p_tf = teacher_forcing_prob(epoch)

    for i in range(num_sequences):
        optimizer.zero_grad()
        x_seq = all_inputs[i].to(device)    # (T,1,2)
        y_seq = all_targets[i].to(device)   # (T,1,2)
        h0 = c0_list[i]

        loss = rollout_loss(x_seq, y_seq, h0, p_tf=p_tf)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(c0_list.parameters()), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % print_every == 0:
        avg = total_loss / num_sequences
        print(f"Epoch {epoch+1:5d} | Avg Loss {avg:.6f} | p_tf {p_tf:.3f}")


# =========================================================
# 4) Plot helpers
# =========================================================
def draw_direction_arrow(ax, data, color, interval=30):
    for i in range(0, len(data) - 2, interval):
        p1 = data[i]
        p2 = data[i + 1]
        ax.annotate(
            "",
            xy=(p2[0], p2[1]),
            xytext=(p1[0], p1[1]),
            arrowprops=dict(arrowstyle="->", color=color, lw=2, mutation_scale=12),
        )


def pca_2d(X):
    """
    X: (N,H)
    return Z: (N,2)
    """
    X = X - X.mean(axis=0, keepdims=True)
    # SVD PCA
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Z = X @ Vt[:2].T
    return Z


# =========================================================
# 5) Rollout + Visualization
#    - Output space: predicted trajectory
#    - Hidden space: PCA( hidden states ) to 2D
# =========================================================
model.eval()

colors = ["#7b2cbf", "#3a0ca3", "#2d6a4f"]

plt.figure(figsize=(14, 6))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

with torch.no_grad():
    for i in range(num_sequences):
        h = c0_list[i].detach().clone()  # (1,1,H)

        # Start from true x0, then free-run
        x0 = all_inputs[i][0:1].to(device)  # (1,1,2)
        x_in = x0.clone()

        preds = [x_in.squeeze().cpu().numpy().copy()]
        states = [h.squeeze().cpu().numpy().copy()]

        rollout_steps = 1500  # try longer to see if it settles into a stable cycle
        for _ in range(rollout_steps):
            y_pred, h = model.forward_step(x_in, h)
            x_in = y_pred  # IMPORTANT: free run (no teacher forcing at test time)

            preds.append(x_in.squeeze().cpu().numpy().copy())
            states.append(h.squeeze().cpu().numpy().copy())

        preds = np.array(preds)    # (N,2)
        states = np.array(states)  # (N,H)

        # Output space
        ax1.plot(preds[:, 0], preds[:, 1], color=colors[i % len(colors)], alpha=0.75, label=f"Seq {i}")
        draw_direction_arrow(ax1, preds, colors[i % len(colors)], interval=40)

        # Hidden space -> PCA 2D
        states_2d = pca_2d(states)
        ax2.plot(states_2d[:, 0], states_2d[:, 1], color=colors[i % len(colors)], alpha=0.70, label=f"Seq {i}")
        draw_direction_arrow(ax2, states_2d, colors[i % len(colors)], interval=40)

# Pretty
ax1.set_title("Output Space (free-running rollout)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.axis("equal")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

ax2.set_title("Hidden Space (PCA to 2D)")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()


# =========================================================
# 6) (Optional) Compare: Ground truth vs prediction overlay
# =========================================================
gt = raw_seqs[0]
plt.figure(figsize=(7, 7))
plt.plot(gt[:, 0], gt[:, 1], lw=2, alpha=0.7, label="Ground Truth")

# use the first len(gt) points from preds
pred_short = preds[: len(gt)]
plt.plot(pred_short[:, 0], pred_short[:, 1], lw=2, alpha=0.7, label="Prediction (free-run)")

plt.title("Ground Truth vs Prediction (overlay)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()