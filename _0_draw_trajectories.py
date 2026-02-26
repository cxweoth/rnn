import os
import matplotlib.pyplot as plt

from fig_utils.draw_utils import draw_direction_arrow
from data_gen.data_reader import read_three_nearby_circle_data, read_two_seperate_circle_data, read_eight_shape_data, read_two_nearby_circle_data

image_path = os.path.join("images", "_0_draw_trajectories")
os.makedirs(image_path, exist_ok=True)


_3_nearby_raw_data = read_three_nearby_circle_data()
_2_seperate_raw_data = read_two_seperate_circle_data()
_8_shape_raw_data = read_eight_shape_data()
_2_nearby_circle_data = read_two_nearby_circle_data()


plt.figure(figsize=(18, 5))

ax0 = plt.subplot(1, 4, 1)
ax1 = plt.subplot(1, 4, 2)
ax2 = plt.subplot(1, 4, 3)
ax3 = plt.subplot(1, 4, 4)

colors = ['#0072E3', '#FF00FF', '#FF0000']

for i in range(len(_3_nearby_raw_data)):
    _3_nearby = _3_nearby_raw_data[i]
    # -------------------------
    # Teacher
    # -------------------------
    ax0.plot(
        _3_nearby[:, 0],
        _3_nearby[:, 1],
        color=colors[i],
        alpha=0.7,
        label=f"$Trajectory^{i}$"
    )
    draw_direction_arrow(ax0, _3_nearby, colors[i])

for i in range(len(_2_seperate_raw_data)):
    _2_seoerate = _2_seperate_raw_data[i]
    # -------------------------
    # Teacher
    # -------------------------
    ax1.plot(
        _2_seoerate[:, 0],
        _2_seoerate[:, 1],
        color=colors[i],
        alpha=0.7,
        label=f"$Trajectory^{i}$"
    )
    draw_direction_arrow(ax1, _2_seoerate, colors[i])

for i in range(len(_8_shape_raw_data)):
    _8_shape = _8_shape_raw_data[i]
    # -------------------------
    # Teacher
    # -------------------------
    ax2.plot(
        _8_shape[:, 0],
        _8_shape[:, 1],
        color=colors[i],
        alpha=0.7,
        label=f"$Trajectory^{i}$"
    )
    draw_direction_arrow(ax2, _8_shape, colors[i])

for i in range(len(_2_nearby_circle_data)):
    _2_nearby_circle = _2_nearby_circle_data[i]
    # -------------------------
    # Teacher
    # -------------------------
    ax3.plot(
        _2_nearby_circle[:, 0],
        _2_nearby_circle[:, 1],
        color=colors[i],
        alpha=0.7,
        label=f"$Trajectory^{i}$"
    )
    draw_direction_arrow(ax3, _2_nearby_circle, colors[i])

ax0.set_title("3 Nearby Trajectories")
ax0.set_xlabel("$x^0$")
ax0.set_ylabel("$x^1$")
ax0.axis('equal')
ax0.grid(True, linestyle='--', alpha=0.5)
ax0.legend()

ax1.set_title("2 Seperate Trajectories")
ax1.set_xlabel("$x^0$")
ax1.set_ylabel("$x^1$")
ax1.axis('equal')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

ax2.set_title("8 Shape Trajectory")
ax2.set_xlabel("$x^0$")
ax2.set_ylabel("$x^1$")
ax2.axis('equal')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

ax3.set_title("2 Nearby Opposite Trajectories ")
ax3.set_xlabel("$x^0$")
ax3.set_ylabel("$x^1$")
ax3.axis('equal')
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend()

plt.tight_layout()
plt.savefig(os.path.join(image_path, "true_trajectories.png"), dpi=300)
plt.close()
