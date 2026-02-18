import os
import json
import numpy as np


def read_three_nearby_circle_data():

    num_sq = 3
    offsets = [(0.0,0.0),(0.11,0.2),(-0.09,-0.2)]
    scales = [1.0,0.7,1.6]

    base_raw = read_one_circle_data()

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


def read_two_seperate_circle_data():

    offsets = [(0.0, 0.0), (0.4, 0.0)]

    base_raw = read_one_circle_data()
    base = np.array([[p["x"], p["y"]] for p in base_raw],
                    dtype=np.float32)

    sequences = []
    for dx, dy in offsets:
        shifted = base + np.array([[dx, dy]], dtype=np.float32)
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



