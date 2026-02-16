import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ===== 讀取 JSON =====
with open("data_gen_by_gesture/one_circle.json", "r") as f:
    raw = json.load(f)

print(len(raw))
data = np.array([[p["x"], p["y"]] for p in raw[100:3500]], dtype=np.float32)

# ===== 建立畫布 =====
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-', linewidth=2)
point, = ax.plot([], [], 'bo', markersize=6)

ax.set_xlim(-0.5, 1.0)
ax.set_ylim(-0.5, 1.0)
ax.set_aspect('equal')
ax.set_title("Mouse Gesture Playback")

# ===== 初始化 =====
def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

# ===== 更新 =====
def update(frame):
    line.set_data(data[:frame, 0], data[:frame, 1])
    point.set_data([data[frame, 0]], [data[frame, 1]])  # ← 修正這裡
    return line, point

ani = FuncAnimation(
    fig,
    update,
    frames=len(data),
    init_func=init,
    interval=10,
    blit=True
)

plt.show()