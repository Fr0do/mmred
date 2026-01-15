"""Visualization functions for MMRed sequences.

This module provides functions to render sequences as images or GIFs
from either DataFrame or JSON-serialized format.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

from ..config import DEFAULT_ROOMS, DEFAULT_CHARS
from ..const import COLORS, WIDTH, HEIGHT


def seq2video(seq, video_path):
    """Render a sequence DataFrame to PNG frames in a directory.
    
    Args:
        seq: pandas DataFrame with character columns and room values
        video_path: Path to output directory
    """
    video_path = Path(video_path)
    video_path.mkdir(parents=True, exist_ok=True)
    for i, row in seq.iterrows():
        frame2png(row, i + 1, video_path)


def frame2png(row, frame_number, video_path, rooms=None, chars=None, colors=None):
    """Render a single frame to PNG.
    
    Args:
        row: pandas Series or dict mapping characters to rooms
        frame_number: 1-indexed frame number
        video_path: Path to output directory
        rooms: List of room names (default: DEFAULT_ROOMS)
        chars: List of character names (default: DEFAULT_CHARS)
        colors: List of RGB tuples for characters (default: COLORS)
    """
    rooms = rooms or DEFAULT_ROOMS
    chars = chars or DEFAULT_CHARS
    colors = colors or COLORS
    
    video_path = Path(video_path)
    
    # Define the grid size and labels
    rows_count, cols = 2, 3
    labels = rooms

    # Character settings: name, color (RGB), relative position (x, y)
    positions = [(0.25, 0.8), (0.75, 0.8), (0.25, 0.25), (0.75, 0.25), (0.5, 0.5)]
    characters = [
        (char, color, position)
        for char, color, position in zip(chars, colors, positions)
    ]

    # Allocation of squares for each character
    if hasattr(row, 'values') and hasattr(row.values, '__call__') == False:
        # pandas Series
        allocations = row.values.tolist()
    elif isinstance(row, dict):
        # dict from JSON (character -> room)
        allocations = [row.get(char, rooms[0]) for char in chars]
    else:
        # Fallback for other iterables
        allocations = list(row)

    # Create the figure with the desired size (WIDTH x HEIGHT pixels)
    fig, ax = plt.subplots(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)

    # Draw the grid of rectangles
    padding = 0.1  # Space between rectangles
    rect_width = 1  # Width of each rectangle
    rect_height = 1.5  # Height of each rectangle to take more vertical space

    # Store the bottom-left rectangle's coordinates for frame number positioning
    bottom_left_x = bottom_left_y = 0

    for i in range(rows_count):
        for j in range(cols):
            # Calculate position
            x = j * (rect_width + padding)
            y = (rows_count - 1 - i) * (rect_height + padding)

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
            if i == (rows_count - 1) and j == 0:
                bottom_left_x, bottom_left_y = x, y

    # Place characters according to the allocations
    for character, square_name in zip(characters, allocations):
        name, color, relative_pos = character

        # Find the square position for the allocated square name
        square_index = labels.index(square_name)
        col = square_index % cols
        row_idx = square_index // cols

        x = col * (rect_width + padding)
        y = (rows_count - 1 - row_idx) * (rect_height + padding)

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
    ax.set_ylim(-0.2 * padding, rows_count * (rect_height + padding) - 0.8 * padding)
    ax.set_aspect("auto")
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(video_path / f"frame_{frame_number:04d}.png")
    plt.close(fig)


def render_sequence_from_json(
    sequence: list[dict[str, Any]],
    output_path: str | Path,
    as_gif: bool = False,
    rooms: list[str] = None,
    chars: list[str] = None,
    colors: list[tuple[int, int, int]] = None,
) -> None:
    """Render a JSON-serialized sequence to images or GIF.
    
    Args:
        sequence: List of step dictionaries with 'step_id' and 'rooms' keys
        output_path: Output path (directory for PNGs, file for GIF)
        as_gif: If True, create a single GIF; otherwise create PNG frames
        rooms: List of room names
        chars: List of character names
        colors: List of RGB tuples for characters
    """
    rooms = rooms or DEFAULT_ROOMS
    chars = chars or DEFAULT_CHARS
    colors = colors or COLORS
    
    output_path = Path(output_path)
    
    # Convert JSON sequence to row format (character -> room mapping)
    frames = []
    for step in sequence:
        char_to_room = {}
        for room, char_list in step["rooms"].items():
            for char in char_list:
                char_to_room[char] = room
        frames.append(char_to_room)
    
    if as_gif:
        if not HAS_IMAGEIO:
            raise ImportError("imageio is required for GIF output. Install with: pip install imageio")
        
        # Create temporary directory for frames
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for i, frame in enumerate(frames, start=1):
                frame2png(frame, i, temp_path, rooms, chars, colors)
            
            # Combine into GIF
            images = []
            for i in range(1, len(frames) + 1):
                img_path = temp_path / f"frame_{i:04d}.png"
                images.append(imageio.imread(img_path))
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(output_path), images, duration=0.5, loop=0)
    else:
        # Render as individual PNGs
        output_path.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames, start=1):
            frame2png(frame, i, output_path, rooms, chars, colors)
