import os
import json
import numpy as np
import matplotlib.pyplot as plt

from thnn.tensor import tensor
from thnn.rnns import RNN_2D_Customized_Hidden_Space
from data_gen.data_reader import read_two_seperate_circle_data
from fig_utils.draw_utils import draw_direction_arrow


# =========================================================
# 0. Helpers
# =========================================================
def lerp_c0(c0_a: tensor, c0_b: tensor, alpha: float) -> tensor:
    a = np.asarray(c0_a.data, dtype=np.float32)
    b = np.asarray(c0_b.data, dtype=np.float32)
    c = (1.0 - alpha) * a + alpha * b
    return tensor(c, requires_grad=False)


def ensure_c0_shape(c0: tensor) -> tensor:
    arr = np.asarray(c0.data)
    if arr.ndim == 3:
        assert arr.shape[0] == 1
        c0 = c0[0]
    return c0


def free_run_random_start(model, c0, steps=1000):
    """
    Pure autonomous free-running from a random start point.
    """
    c0 = ensure_c0_shape(c0)

    # 隨機起始 input
    x_rand = np.random.uniform(-1.5, 1.5, size=(1,1,2)).astype(np.float32)
    x0 = tensor(x_rand, requires_grad=False)

    h = c0
    x_t = x0[0]

    traj = np.zeros((steps, 2), dtype=np.float32)

    for t in range(steps):
        h = model.act(model.ih(x_t) + model.hh(h))
        y = model.fc(h)

        traj[t] = y.data[0]
        x_t = y

    return traj


# =========================================================
# 1. Read teacher data
# =========================================================
raw_seqs = read_two_seperate_circle_data()
assert len(raw_seqs) >= 2

true0 = np.asarray(raw_seqs[0], dtype=np.float32)
true1 = np.asarray(raw_seqs[1], dtype=np.float32)


# =========================================================
# 2. Load model + state
# =========================================================
hidden_dim = 3
model = RNN_2D_Customized_Hidden_Space(hidden_dim)

image_path = os.path.join("images", "_2_circle_seperate", "tf_adam")
dict_path = os.path.join(image_path, "state_dict.json")

with open(dict_path, "r") as f:
    state = json.load(f)

model.load_state_dict(state)
print("Model weights restored successfully.")

assert "c0_list" in state and len(state["c0_list"]) >= 2

c0_a = tensor(np.array(state["c0_list"][0], dtype=np.float32), requires_grad=False)
c0_b = tensor(np.array(state["c0_list"][1], dtype=np.float32), requires_grad=False)


# =========================================================
# 3. Plot 10 interpolations (fixed random start)
# =========================================================
rows, cols = 2, 5
n_plots = rows * cols
alphas = np.linspace(0.0, 1.0, n_plots)

fig, axes = plt.subplots(rows, cols, figsize=(14, 6))
axes = np.asarray(axes)

steps = 1000
legend_handles = None

# -------- 只 random 一次 --------
x_rand = np.random.uniform(-1.5, 1.5, size=(1,1,2)).astype(np.float32)
x0 = tensor(x_rand, requires_grad=False)

for idx, alpha in enumerate(alphas):
    ax = axes[idx // cols, idx % cols]

    # ---------- Draw teacher ----------
    h_true0, = ax.plot(true0[:, 0], true0[:, 1], linewidth=1.2, alpha=0.3)
    draw_direction_arrow(ax, true0, color=h_true0.get_color())

    h_true1, = ax.plot(true1[:, 0], true1[:, 1], linewidth=1.2, alpha=0.3)
    draw_direction_arrow(ax, true1, color=h_true1.get_color())

    # ---------- Interpolated hidden ----------
    c0 = lerp_c0(c0_a, c0_b, float(alpha))
    c0 = ensure_c0_shape(c0)

    # ---------- Free-run ----------
    h = c0
    x_t = x0[0]

    traj = np.zeros((steps, 2), dtype=np.float32)

    for t in range(steps):
        h = model.act(model.ih(x_t) + model.hh(h))
        y = model.fc(h)

        traj[t] = y.data[0]
        x_t = y

    h_pred, = ax.plot(traj[:, 0], traj[:, 1], linewidth=1.5)
    draw_direction_arrow(ax, traj, color=h_pred.get_color())

    ax.set_title(f"c0 interp α={alpha:.2f}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    if legend_handles is None:
        legend_handles = [
            (h_true0, "True #0 (teacher)"),
            (h_true1, "True #1 (teacher)"),
            (h_pred,  "Pred (fixed random start)")
        ]

# -------- legend --------
handles = [h for (h, _) in legend_handles]
labels  = [l for (_, l) in legend_handles]

fig.legend(handles, labels,
           loc="upper center",
           ncol=2,
           frameon=False,
           bbox_to_anchor=(0.5, 1.08))

plt.tight_layout()

os.makedirs(image_path, exist_ok=True)
out_path = os.path.join(image_path,
                        "c0_interp_10_fixed_random_start.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {out_path}")