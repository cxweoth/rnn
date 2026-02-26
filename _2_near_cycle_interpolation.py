import os
import json
import numpy as np
import matplotlib.pyplot as plt

from thnn.tensor import tensor
from thnn.rnns import RNN_2D_Customized_Hidden_Space
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


def pick_random_x0(radius=1.0, seed=0) -> tensor:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    arr = np.array([[[x, y]]], dtype=np.float32)  # (1,1,2)
    return tensor(arr, requires_grad=False)


def free_run_single_traj(model, c0, x0, steps=1000):
    c0 = ensure_c0_shape(c0)

    x_t = x0[0]   # (B,2)
    h = c0

    traj = np.zeros((steps, 2), dtype=np.float32)

    for t in range(steps):
        h = model.act(model.ih(x_t) + model.hh(h))
        y = model.fc(h)

        traj[t, :] = y.data[0]
        x_t = y

    return traj


# =========================================================
# 1. Load model + state
# =========================================================
hidden_dim = 16
model = RNN_2D_Customized_Hidden_Space(hidden_dim)

image_path = os.path.join("images", "_2_near_opposite_cycle", "tf_good")
dict_path = os.path.join(image_path, "state_dict.json")

with open(dict_path, "r") as f:
    state = json.load(f)

model.load_state_dict(state)
print("Model weights restored successfully.")

assert "c0_list" in state and len(state["c0_list"]) >= 2

c0_a = tensor(np.array(state["c0_list"][0], dtype=np.float32), requires_grad=False)
c0_b = tensor(np.array(state["c0_list"][1], dtype=np.float32), requires_grad=False)

c0_a = ensure_c0_shape(c0_a)
c0_b = ensure_c0_shape(c0_b)

# 任意起始點
x0 = pick_random_x0(radius=1.0, seed=0)

# =========================================================
# 2. Plot interpolations (2 rows × 5 cols => 10 plots)
# =========================================================
rows, cols = 2, 5
n_plots = rows * cols                   # 10
alphas = np.linspace(0.0, 1.0, n_plots) # 10點 interpolation

fig, axes = plt.subplots(rows, cols, figsize=(14, 6))
axes = np.asarray(axes)

steps = 1000

plot_idx = 0
for r in range(rows):
    for c in range(cols):
        ax = axes[r, c]

        alpha = float(alphas[plot_idx])

        # ---- interpolate hidden initial state ----
        c0 = lerp_c0(c0_a, c0_b, alpha)

        # ---- free-running trajectory ----
        traj = free_run_single_traj(model, c0, x0, steps=steps)

        # ---- plot trajectory ----
        line, = ax.plot(traj[:, 0], traj[:, 1], linewidth=1.2)

        # ---- draw direction arrow ----
        draw_direction_arrow(ax, traj, color=line.get_color())

        # ---- alpha label ----
        ax.set_title(f"α = {alpha:.2f}", fontsize=10)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")

        plot_idx += 1

plt.tight_layout()

os.makedirs(image_path, exist_ok=True)
out_path = os.path.join(image_path, "c0_interp_grid_2x5_single_line.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {out_path}")
