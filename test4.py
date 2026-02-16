import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data_gen_by_gesture.data_reader import read_one_circle_data
from data_gen.circle_gen import generate_two_circle_sequence


# ---------------------------------------------------------
# 1. 根據你的資料生成三條序列
# ---------------------------------------------------------
def generate_teacher_like_data():

    offsets = [(0.0, 0.0), (0.7, 0.0)]

    base_raw = read_one_circle_data()

    base = np.array([[p["x"], p["y"]] for p in base_raw],
                    dtype=np.float32)

    sequences = []
    for dx, dy in offsets:
        shifted = base + np.array([[dx, dy]], dtype=np.float32)
        sequences.append(shifted)

    return sequences

import matplotlib.pyplot as plt
import numpy as np

def draw_direction_arrow(ax, data, color, interval=20):
    """
    每隔 interval 個點畫一個方向箭頭
    data shape: (T,2)
    """
    for i in range(0, len(data) - 2, interval):
        p1 = data[i]
        p2 = data[i + 1]
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

# 生成資料
raw_seqs = generate_two_circle_sequence()

# 畫圖
plt.figure(figsize=(6,6))
colors = ['#7b2cbf', '#3a0ca3', '#2d6a4f']

for i, seq in enumerate(raw_seqs):
    seq = np.array(seq)
    plt.plot(seq[:,0], seq[:,1],
             color=colors[i],
             alpha=0.8,
             label=f'Seq {i}')
    
    draw_direction_arrow(plt.gca(), seq, colors[i], interval=15)

plt.title("Teacher-like Sequences with Direction Arrows")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.show()