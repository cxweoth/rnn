def draw_direction_arrow(ax, data, color, interval=30):
    for i in range(0, len(data)-2, interval):
        p1 = data[i]
        p2 = data[i+1]
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