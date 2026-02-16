import numpy as np

def generate_circle_sequence(n_points=100, radius=1.0):
    """
    Generate circle sequence aligned with read_one_circle_data interface.

    Returns:
        list of dict:
        [
            {"x": float, "y": float},
            ...
        ]
    """

    theta = np.linspace(0, 2*np.pi, n_points)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    data = []

    for i in range(n_points):

        data.append({
            "x": float(x[i]),
            "y": float(y[i])
        })

    return data

def generate_two_circle_sequence(n_points=120):

    sequences = []
    # 定義三組不同的參數 (半徑x, 半徑y, 中心x, 中心y)
    params = [
        (0.8, 0.9, 0.1, 0.0),   # 序列 0: 稍微扁平且往右偏
        (1.0, 1.0, 0.3, 0.0),   # 序列 1: 標準單位圓
    ]
    
    for rx, ry, ox, oy in params:
        # 隨機起始相位，讓 $c_0$ 學習不同的進入點
        start_phase = np.random.uniform(0, 2 * np.pi)
        theta = np.linspace(start_phase, start_phase + 2 * np.pi, n_points)
        x = rx * np.cos(theta) + ox
        y = ry * np.sin(theta) + oy
        sequences.append(np.column_stack((x, y)).astype(np.float32))
    return sequences

def generate_two_irregular_circles(n_points=300):

    sequences = []
    
    params = [
        (1.0, 0.0),   # 圓1中心
        (2.5, 0.0),   # 圓2中心
    ]
    
    for ox, oy in params:
        
        theta = np.zeros(n_points)
        velocity = np.random.uniform(0.04, 0.07)

        # 不規則角速度
        for t in range(1, n_points):
            velocity += np.random.uniform(-0.01, 0.01)
            velocity = np.clip(velocity, 0.02, 0.12)
            theta[t] = theta[t-1] + velocity

        # 半徑輕微變形
        r = 1.0 + 0.2*np.sin(3*theta)

        x = r * np.cos(theta) + ox
        y = r * np.sin(theta) + oy

        sequences.append(np.column_stack((x, y)).astype(np.float32))

    return sequences

# import matplotlib.pyplot as plt

# raw_data = generate_two_irregular_circles(200)

# plt.figure(figsize=(6, 6))

# for i, seq in enumerate(raw_data):
#     plt.plot(seq[:, 0], seq[:, 1], label=f"Circle {i}")

# plt.title("Two Generated Circle Sequences")
# plt.xlabel("x0")
# plt.ylabel("x1")
# plt.axis('equal')   # 很重要
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()

# plt.show()


