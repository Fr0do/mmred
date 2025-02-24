import matplotlib.pyplot as plt

from qgen.const import ROOMS, CHARS, COLORS, WIDTH, HEIGHT


def seq2video(seq, video_path):
    video_path.mkdir(parents=True, exist_ok=True)
    for i, row in seq.iterrows():
        frame2png(row, i + 1, video_path)


def frame2png(row, frame_number, video_path):
    # Define the grid size and labels
    rows, cols = 2, 3
    labels = ROOMS

    # Character settings: name, color (RGB), relative position (x, y)
    positions = [(0.25, 0.8), (0.75, 0.8), (0.25, 0.25), (0.75, 0.25), (0.5, 0.5)]
    characters = [
        (char, color, position)
        for char, color, position in zip(CHARS, COLORS, positions)
    ]

    # Allocation of squares for each character
    allocations = row.values.tolist()

    # Create the figure with the desired size (WIDTH x HEIGHT pixels)
    fig, ax = plt.subplots(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)

    # Draw the grid of rectangles
    padding = 0.1  # Space between rectangles
    rect_width = 1  # Width of each rectangle
    rect_height = 1.5  # Height of each rectangle to take more vertical space

    # Store the bottom-left rectangle's coordinates for frame number positioning
    bottom_left_x = bottom_left_y = 0

    for i in range(rows):
        for j in range(cols):
            # Calculate position
            x = j * (rect_width + padding)
            y = (rows - 1 - i) * (rect_height + padding)

            # Draw rectangle
            square = plt.Rectangle(
                (x, y),
                rect_width,
                rect_height,
                fill=None,
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(square)

            # Add label in the bottom-left corner of the rectangle
            label_index = i * cols + j
            plt.text(
                x + 0.02,
                y + 0.02,
                labels[label_index],
                horizontalalignment="left",
                verticalalignment="bottom",
                fontweight="bold",
                fontsize=14,
            )

            # Store coordinates of the bottom-left rectangle
            if i == (rows - 1) and j == 0:
                bottom_left_x, bottom_left_y = x, y

    # Place characters according to the allocations
    for character, square_name in zip(characters, allocations):
        name, color, relative_pos = character

        # Find the square position for the allocated square name
        square_index = labels.index(square_name)
        col = square_index % cols
        row = square_index // cols

        x = col * (rect_width + padding)
        y = (rows - 1 - row) * (rect_height + padding)

        # Calculate character position within the square
        char_x = x + relative_pos[0] * rect_width
        char_y = y + relative_pos[1] * rect_height

        # Plot the character as a dot
        ax.plot(char_x, char_y, "o", color=[c / 255 for c in color], markersize=16)

        # Add the character's name above the dot
        plt.text(
            char_x,
            char_y + 0.08,
            name,
            horizontalalignment="center",
            verticalalignment="bottom",
            fontweight="bold",
            fontsize=12,
        )

    # Add the frame number below the bottom-left rectangle
    plt.text(
        bottom_left_x + rect_width * 1.5 + padding,
        bottom_left_y - 0.08,
        f"Step {frame_number}",
        horizontalalignment="center",
        verticalalignment="top",
        fontweight="bold",
        fontsize=14,
    )

    # Adjust plot limits and aesthetics
    ax.set_xlim(-0.2 * padding, cols * (rect_width + padding) - 0.8 * padding)
    ax.set_ylim(-0.2 * padding, rows * (rect_height + padding) - 0.8 * padding)
    ax.set_aspect("auto")
    ax.axis("off")

    # Show the plot
    # plt.show()
    plt.tight_layout()
    fig.savefig(video_path / f"frame_{frame_number:04d}.png")
    plt.close(fig)
