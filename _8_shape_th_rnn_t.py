"""
Docstring for _8_shape_th_rnn_t
Descriptions: 8 shape within teacher 
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from thnn.tensor import tensor
from thnn.loss import MSELoss
from thnn.optimizer import Adam, SGD
from thnn.utils import rollout_one
from thnn.rnns import RNN_2D_Customized_Hidden_Space

from fig_utils.draw_utils import draw_direction_arrow
from data_gen.data_reader import read_eight_shape_data


# =========================================================
# 0. create image folder
# =========================================================

image_path = os.path.join("images", "_8_shape", "t")
os.makedirs(image_path, exist_ok=True)


# =========================================================
# 1. read three nearby circle data
# =========================================================

raw_seqs = read_eight_shape_data()

num_sequences = len(raw_seqs)
T = raw_seqs[0].shape[0] - 1


# =========================================================
# 2. build inputs 
#    each shape = (T,1,2)
# =========================================================

all_inputs = []
all_targets = []

for seq in raw_seqs:

    # seq = [[x0,y0],[x1,y1],...] T*Dimension=T*2
    # reshape to [[[x0,y0]],[[x1,y1]],...] T*Batch*Dimension=T*1*2
    inp = tensor(seq[:-1].reshape(T,1,2))
    tgt = tensor(seq[1:].reshape(T,1,2))

    # different sequences within different hidden (context) values
    all_inputs.append(inp)
    all_targets.append(tgt)

# =========================================================
# 3. Init model
# =========================================================

hidden_dim = 3
model = RNN_2D_Customized_Hidden_Space(hidden_dim)

# =========================================================
# 4. learnable initial hidden states
# =========================================================

# use these variables to control different trajectories
c0_list = [

    tensor(
        np.random.randn(1,1,hidden_dim).astype(np.float32)*0.1
    )

    for _ in range(num_sequences)
]


# =========================================================
# 5. optimizer and loss
# =========================================================

optimizer = SGD(
    model.parameters()+c0_list, # to learn the c0 initial contexts
    lr=0.01
)

criterion = MSELoss()


# =========================================================
# 6. training loop
# =========================================================

print("Training RNN (correct latent c0 version)...")

epochs = 10000
loss_history = []

for epoch in range(epochs):

    total_loss = 0.0

    for i in range(num_sequences):
        
        # 1) compute forward value
        output, h_final = model(all_inputs[i], c0_list[i])

        # 2) compute loss
        loss = criterion(output, all_targets[i])

        # 3) init optimizer to zero grad (not true zero, it's none)
        optimizer.zero_grad()

        # 4) do the backward propagation
        loss.backward()

        # 5 update one step
        optimizer.step()

        total_loss += float(loss.data)

    avg_loss = total_loss / num_sequences

    loss_history.append(avg_loss)

    if (epoch+1)%200==0:
        print(f"Epoch {epoch+1}, Loss {avg_loss:.6f}")


# =========================================================
# 7. rollout and plot (SAVE ONLY, NO SHOW)
# =========================================================

plt.figure(figsize=(18,5))

ax0 = plt.subplot(1,3,1)
ax1 = plt.subplot(1,3,2)

if hidden_dim == 3:
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

        pca = PCA(n_components=2)
        states_2d = pca.fit_transform(states)

        ax2.plot(
            states_2d[:,0],
            states_2d[:,1],
            color=colors[i],
            alpha=0.6,
            label=f"$c_0^{i}$"
        )

        draw_direction_arrow(ax2, states_2d, colors[i])


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

if hidden_dim != 3:
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