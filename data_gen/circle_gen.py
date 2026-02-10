import numpy as np

def generate_circle_sequence(n_points=100, radius=1.0):
    """
    Generates a sequence of (x, y) coordinates forming a circle.
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    # Shape: (n_points, 2)
    data = np.column_stack((x, y))
    return data

# import matplotlib.pyplot as plt

# # Generate the data using your function
# raw_data = generate_circle_sequence(200)

# print("raw:", raw_data)
# # Extract x and y coordinates
# x0_coords = raw_data[:, 0]
# x1_coords = raw_data[:, 1]
# print("x_0", x0_coords)
# print("x_1", x1_coords)

# # Create the plot
# plt.figure(figsize=(6, 6))
# plt.plot(x0_coords, x1_coords, 'b-', label='Generated Path') # Plot the line
# plt.scatter(x0_coords, x1_coords, c='red', s=10, alpha=0.5, label='Data Points') # Plot individual points

# # Formatting the plot
# plt.title("Generated Circle Sequence for RNN Training")
# plt.xlabel("x0")
# plt.ylabel("x1")
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.axis('equal') # IMPORTANT: Ensures the circle looks round
# plt.legend()

# plt.show()


