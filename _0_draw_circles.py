import os
import matplotlib.pyplot as plt

from fig_utils.draw_utils import draw_direction_arrow
from data_gen.data_reader import read_three_true_circles_sequence, read_three_nearby_circle_one_opposite_data

image_path = os.path.join("images", "_0_draw_circles")
os.makedirs(image_path, exist_ok=True)


_3_circles = read_three_nearby_circle_one_opposite_data()

plt.figure(figsize=(18, 5))

ax0 = plt.subplot(1, 3, 1)
ax1 = plt.subplot(1, 3, 2)
ax2 = plt.subplot(1, 3, 3)

colors = ['#0072E3', '#FF00FF', '#FF0000']

for i in range(len(_3_circles)):
    _circle = _3_circles[i]
    # -------------------------
    # Teacher
    # -------------------------
    ax0.plot(
        _circle[:, 0],
        _circle[:, 1],
        color=colors[i],
        alpha=0.7,
        label=f"$Trajectory^{i}$"
    )
    draw_direction_arrow(ax0, _circle, colors[i])

ax0.set_title("3 Cicles")
ax0.set_xlabel("$x^0$")
ax0.set_ylabel("$x^1$")
ax0.axis('equal')
ax0.grid(True, linestyle='--', alpha=0.5)
ax0.legend()


plt.tight_layout()
plt.savefig(os.path.join(image_path, "true_trajectories.png"), dpi=300)
plt.close()
