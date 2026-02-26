"""
Docstring for _2_near_opposite_cycle_th_rnn_tf
Descriptions: 2 cycle within teacher and free running
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from thnn.tensor import tensor
from thnn.loss import MSELoss
from thnn.optimizer import Adam, SGD, SGD_Momentum
from thnn.utils import rollout_one, clip_grad_norm
from thnn.rnns import RNN_2D_Customized_Hidden_Space

from fig_utils.draw_utils import draw_direction_arrow
from data_gen.data_reader import read_two_nearby_circle_data


# =========================================================
# 0. create image folder
# =========================================================
image_path = os.path.join("images", "_2_near_opposite_cycle", "tf")
os.makedirs(image_path, exist_ok=True)


# =========================================================
# 1. read two separate circle data
# =========================================================
raw_seqs = read_two_nearby_circle_data()

num_sequences = len(raw_seqs)
T = raw_seqs[0].shape[0] - 1


# =========================================================
# 2. build inputs / targets
#    each shape = (T, 1, 2)
# =========================================================
all_inputs = []
all_targets = []

for seq in raw_seqs:
    inp = tensor(seq[:-1].reshape(T, 1, 2), requires_grad=False)
    tgt = tensor(seq[1:].reshape(T, 1, 2), requires_grad=False)
    all_inputs.append(inp)
    all_targets.append(tgt)


# =========================================================
# 3. Init model
# =========================================================
hidden_dim = 16
model = RNN_2D_Customized_Hidden_Space(hidden_dim)


# =========================================================
# 4. learnable initial hidden states (c0)
# =========================================================
c0_list = [
    tensor(np.random.randn(1, 1, hidden_dim).astype(np.float32) * 0.1)
    for _ in range(num_sequences)
]


# =========================================================
# 5. optimizer and loss
# =========================================================
optimizer = Adam(
    model.parameters() + c0_list,
    lr=0.003
)

criterion = MSELoss()


# =========================================================
# 6. training loop (HYBRID scheduled sampling)
# =========================================================
print("Training RNN (hybrid teacher forcing + free running)...")

epochs = 20000
loss_history = []


def rollout_loss(x_seq: tensor, y_seq: tensor, h0: tensor, teacher_forcing_ratio=0.5):
    """
    x_seq: (T, B, 2)  here B=1
    y_seq: (T, B, 2)
    h0  : (1, B, H)   learnable c0
    """

    T = x_seq.data.shape[0]

    # hidden should be (B, H)
    h = h0[0]  # (B, H)

    # initial input should be (B, 2)
    x_t = x_seq[0]  # (B, 2)

    total_loss = None

    for t in range(T):
        # RNN cell update
        h = model.act(model.ih(x_t) + model.hh(h))
        y = model.fc(h)  # (B, 2)

        target_t = y_seq[t]  # (B, 2)
        loss_t = criterion(y, target_t)

        total_loss = loss_t if total_loss is None else (total_loss + loss_t)

        # choose next input
        if (t + 1) < T and (np.random.rand() < teacher_forcing_ratio):
            # teacher forcing: use ground-truth next input
            x_t = x_seq[t + 1]  # (B, 2)
        else:
            # free running: use model output as next input
            # IMPORTANT: do NOT detach here, otherwise you cut BPTT
            x_t = y 

    return total_loss


for epoch in range(epochs):
    total_loss = 0.0

    # scheduled sampling ratio: from 0.8 -> 0.1
    teacher_forcing_ratio = max(0.8 * (1 - epoch / epochs), 0.1)

    for i in range(num_sequences):
        optimizer.zero_grad()

        loss = rollout_loss(
            all_inputs[i],
            all_targets[i],
            c0_list[i],
            teacher_forcing_ratio
        )

        loss.backward()

        clip_grad_norm(model.parameters(), 1.0)  

        optimizer.step()

        total_loss += float(loss.data)

    avg_loss = total_loss / num_sequences
    loss_history.append(avg_loss)

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}, Loss {avg_loss:.6f}, TF_ratio={teacher_forcing_ratio:.3f}")

import json

state_dict = model.state_dict()

dict_path = os.path.join(image_path, "state_dict.json")

c0_serializable = [
    c0.data.tolist()
    for c0 in c0_list
]

state_dict["c0_list"] = c0_serializable

with open(dict_path, "w") as f:
    json.dump(state_dict, f, indent=4)

print(f"State dict saved to {dict_path}")

# =========================================================
# 7. rollout and plot (SAVE ONLY, NO SHOW)
# =========================================================

plt.figure(figsize=(18,5))

ax0 = plt.subplot(1,3,1)
ax1 = plt.subplot(1,3,2)

# 3D hidden 或 >=4D PCA 都要 3D axes
if hidden_dim >= 3:
    ax2 = plt.subplot(1,3,3, projection='3d')
else:
    ax2 = plt.subplot(1,3,3)

colors = ['#0072E3', '#FF00FF', '#FF0000']

for i in range(num_sequences):

    raw_data = raw_seqs[i]

    preds, states = rollout_one(
        model,
        c0_list[i],
        all_inputs[i],
        steps=1000
    )

    # -------------------------
    # Teacher
    # -------------------------
    ax0.plot(
        raw_data[:,0],
        raw_data[:,1],
        color=colors[i],
        alpha=0.7,
        label=f"$raw^{i}$"
    )
    draw_direction_arrow(ax0, raw_data, colors[i])


    # -------------------------
    # Output Space
    # -------------------------
    ax1.plot(
        preds[:,0],
        preds[:,1],
        color=colors[i],
        alpha=0.7,
        label=f"$c_0^{i}$"
    )
    draw_direction_arrow(ax1, preds, colors[i])


    # -------------------------
    # Hidden Space
    # -------------------------

    hd = states.shape[1]

    # ===== 1D =====
    if hd == 1:

        dummy = np.column_stack([
            states[:,0],
            np.zeros_like(states[:,0])
        ])

        ax2.plot(
            dummy[:,0],
            dummy[:,1],
            color=colors[i],
            alpha=0.6,
            label=f"$c_0^{i}$"
        )

        draw_direction_arrow(ax2, dummy, colors[i])

    # ===== 2D =====
    elif hd == 2:

        ax2.plot(
            states[:,0],
            states[:,1],
            color=colors[i],
            alpha=0.6,
            label=f"$c_0^{i}$"
        )

        draw_direction_arrow(ax2, states, colors[i])

    # ===== 3D =====
    elif hd == 3:

        ax2.plot(
            states[:,0],
            states[:,1],
            states[:,2],
            color=colors[i],
            alpha=0.6,
            label=f"$c_0^{i}$"
        )

        draw_direction_arrow(ax2, states, colors[i])

    # ===== >=4D → PCA =====
    else:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=3)
        states_3d = pca.fit_transform(states)   # (T,3)

        ax2.plot(
            states_3d[:, 0],
            states_3d[:, 1],
            states_3d[:, 2],
            color=colors[i],
            alpha=0.6,
            label=f"$c_0^{i}$"
        )

        # draw arrow on PC1-PC2 plane (keep your arrow util unchanged)
        draw_direction_arrow(ax2, states_3d[:, :2], colors[i])


# -------------------------
# Formatting
# -------------------------

ax0.set_title("True Trajectory")
ax0.set_xlabel("$x^0$")
ax0.set_ylabel("$x^1$")
ax0.axis('equal')
ax0.grid(True, linestyle='--', alpha=0.5)
ax0.legend()

ax1.set_title("Output Space ($x^1$ vs $x^0$)")
ax1.set_xlabel("$x^0$")
ax1.set_ylabel("$x^1$")
ax1.axis('equal')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

ax2.set_title("Internal State Space")

if hidden_dim == 1:
    ax2.set_xlabel("$c^0$")
    ax2.set_ylabel("(dummy)")

elif hidden_dim == 2:
    ax2.set_xlabel("$c^0$")
    ax2.set_ylabel("$c^1$")

elif hidden_dim == 3:
    ax2.set_xlabel("$c^0$")
    ax2.set_ylabel("$c^1$")
    ax2.set_zlabel("$c^2$")

else:
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")

if hidden_dim in [1, 2]:
    ax2.axis('equal')

ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(image_path, "trajectories.png"),
    dpi=300
)

plt.close()


# =========================================================
# Plot average loss
# =========================================================

plt.figure(figsize=(6,4))

plt.plot(loss_history, color="#0072E3", linewidth=2)

plt.title("Average Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")

plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

plt.savefig(
    os.path.join(image_path, "loss_curve.png"),
    dpi=300
)

plt.close()