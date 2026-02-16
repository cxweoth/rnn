import matplotlib.pyplot as plt
import json

positions = []

fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Draw with mouse (close window to save JSON)")

def on_move(event):
    if event.xdata is not None and event.ydata is not None:
        positions.append({
            "x": float(event.xdata),
            "y": float(event.ydata)
        })
        xs = [p["x"] for p in positions]
        ys = [p["y"] for p in positions]
        line.set_data(xs, ys)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.show()

# 儲存 JSON
with open("mouse_sequence.json", "w") as f:
    json.dump(positions, f, indent=2)

print("Saved", len(positions), "points to mouse_sequence.json")