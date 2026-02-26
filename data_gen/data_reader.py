import os
import json
import numpy as np


def read_three_nearby_circle_data():

    num_sq = 3
    offsets = [
        (0.0,0.0),
        (0.01,0.01),
        (-0.02,-0.02)
    ]
    scales = [1.0,0.99,1.01]

    base_raw = read_one_circle_data()[:50]

    base = np.array(
        [[p["x"],p["y"]] for p in base_raw],
        dtype=np.float32
    )

    sequences=[]

    for i in range(num_sq):
        
        dx, dy = offsets[i]
        scale = scales[i]
        scaled_shifted = scale*(base + np.array([[dx,dy]],dtype=np.float32))

        sequences.append(scaled_shifted)

    return sequences

def read_three_nearby_circle_one_opposite_data():

    num_sq = 3

    offsets = [
        (0.0,0.0),
        (0.01,0.01),
        (-0.02,-0.02)
    ]

    scales = [1.0,1.0,1.0]

    base_raw = read_one_circle_data()[:50]

    base = np.array(
        [[p["x"],p["y"]] for p in base_raw],
        dtype=np.float32
    )

    sequences=[]

    for i in range(num_sq):
        
        dx, dy = offsets[i]
        scale = scales[i]

        scaled_shifted = scale * (
            base + np.array([[dx,dy]],dtype=np.float32)
        )

        # 🔥 make the third one opposite direction
        if i == 2:
            scaled_shifted = scaled_shifted[::-1]

        sequences.append(scaled_shifted)

    return sequences

def read_two_nearby_circle_data():

    offsets = [(0.0, 0.0), (-0.5, -0.01)]

    base_raw = read_one_circle_data()
    base = np.array([[p["x"], p["y"]] for p in base_raw],
                    dtype=np.float32)

    sequences = []

    for i in range(len(offsets)):
        dx, dy = offsets[i]
        shifted = base + np.array([[dx, dy]], dtype=np.float32)

        if i == 1:
            # 左右鏡射（對 y 軸）
            shifted[:, 0] = -shifted[:, 0]

        sequences.append(shifted)

    return sequences

def read_two_seperate_circle_data():

    offsets = [(0.0, 0.0), (0.4, 0.0)]

    base_raw = read_one_circle_data()
    base = np.array([[p["x"], p["y"]] for p in base_raw],
                    dtype=np.float32)

    sequences = []
    for i in range(len(offsets)):
        dx, dy = offsets[i]
        shifted = base + np.array([[dx, dy]], dtype=np.float32)
        if i == 1:
            shifted = shifted[::-1]
        sequences.append(shifted)

    return sequences


def read_one_circle_data():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, "one_circle.json")

    with open(data_path, "r") as f:
        raw = json.load(f)

    filter_data = raw[100:1100:10]

    return filter_data


def read_eight_shape_data():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, "eight_shape.json")

    with open(data_path, "r") as f:
        raw = json.load(f)

    filter_data = raw[100:1500:10]

    base = np.array([[p["x"], p["y"]] for p in filter_data], dtype=np.float32)  # (T,2)
    return [base]


def read_three_true_circles_sequence(n_points=100):

    sequences = []
    # 定義三組不同的參數 (半徑x, 半徑y, 中心x, 中心y)
    params = [
        (1.0, 1.0, 0.0, 0.0),   
        (0.95, 0.95, 0.0, 0.0),   
        (1.05, 1.05, 0.0, 0.0),   
    ]
    
    for rx, ry, ox, oy in params:
        # 隨機起始相位，讓 $c_0$ 學習不同的進入點
        start_phase = np.random.uniform(0, 2 * np.pi)
        theta = np.linspace(start_phase, start_phase + 2 * np.pi, n_points)
        x = rx * np.cos(theta) + ox
        y = ry * np.sin(theta) + oy
        sequences.append(np.column_stack((x, y)).astype(np.float32))
    return sequences
