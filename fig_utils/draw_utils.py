def draw_direction_arrow(ax, data, color, interval=30):

    dim = data.shape[1]

    for i in range(0, len(data)-2, interval):

        p1 = data[i]
        p2 = data[i+1]

        # =========================
        # 2D case
        # =========================
        if dim == 2:

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

        # =========================
        # 3D case
        # =========================
        elif dim == 3:

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dz = p2[2] - p1[2]

            # normalize arrow length
            norm = (dx**2 + dy**2 + dz**2) ** 0.5 + 1e-8
            dx /= norm
            dy /= norm
            dz /= norm

            scale = 0.2  # 控制箭頭長度

            ax.quiver(
                p1[0], p1[1], p1[2],
                dx*scale, dy*scale, dz*scale,
                color=color,
                linewidth=2,
                arrow_length_ratio=0.3
            )