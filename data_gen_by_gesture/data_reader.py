import os
import json


def read_one_circle_data():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, "one_circle.json")

    with open(data_path, "r") as f:
        raw = json.load(f)

    filter_data = raw[100:1000:10]

    return filter_data


def read_eight_shape_data():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, "eight_shape.json")

    with open(data_path, "r") as f:
        raw = json.load(f)

    filter_data = raw[100:1500:10]

    return filter_data



