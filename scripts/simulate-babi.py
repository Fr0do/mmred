import os
import random
import csv
import pygame
from multiprocessing import Pool
from datetime import datetime
import math

###############################################################################
# Enable headless mode by using dummy video driver before initializing Pygame #
###############################################################################
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Set a global random seed for reproducible results
random.seed(1337)

################################################################################
# Adaptive parameters: we define a default resolution and derive sizes from it #
################################################################################
DEFAULT_WIDTH, DEFAULT_HEIGHT = 512, 512

def init_pygame(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    """
    Initialize pygame in headless mode (no real window).
    Returns a display surface where we can draw.
    """
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Character Movement Simulation")
    return screen

def generate_room_layout(room_names, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    """
    Generate a dict of rooms in a grid layout:
    {
      "kitchen": pygame.Rect(...),
      "bathroom": pygame.Rect(...),
      ...
    }
    """
    cols = 3  # number of columns in the grid
    margin = math.sqrt(DEFAULT_WIDTH * DEFAULT_HEIGHT) // 10
    room_width = (width - (cols + 1) * margin) // cols
    room_height = (height - (len(room_names) // cols + 1) * margin) // (len(room_names) // cols)

    layout = {}
    for i, room_name in enumerate(room_names):
        row = i // cols
        col = i % cols
        x = margin + col * (room_width + margin)
        y = margin + row * (room_height + margin)
        layout[room_name] = pygame.Rect(x, y, room_width, room_height)

    return layout

def draw_environment(
    screen, rooms, bins, characters, step_num, folder,
    width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT
):
    """
    Draw rooms and characters, then save the image as step_XX.png
    """
    # Define colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    # Fill the background
    screen.fill(WHITE)

    # Adapt font and circle sizes to the resolution
    font_size_rooms = int(min(width, height) // 30)      # for room labels
    font_size_chars = int(min(width, height) // 34)      # for character labels
    circle_radius   = int(min(width, height) // 40)
    text_offset     = 1.75 * circle_radius  # how far above the circle text will appear

    # More crisp fonts (SysFont with bold=True if desired)
    font_room = pygame.font.Font("fonts/ARIALBD.TTF", font_size_rooms)
    font_char = pygame.font.Font("fonts/ARIAL.TTF", font_size_chars)

    # Draw rooms
    for room, rect in rooms.items():
        pygame.draw.rect(screen, BLACK, rect, 2)
        text = font_room.render(room.capitalize(), True, BLACK)
        # Place the room name near the top-left corner of the room
        screen.blit(text, (rect.x + 5, rect.y + rect.height - 1.5 * font_size_rooms))

    # Draw characters (circles + labels)
    for char, data in characters.items():
        room_name = data["room"]
        if room_name.lower() not in bins:
            continue

        # Identify bin position
        room_bins = bins[room_name.lower()]
        char_index = list(characters.keys()).index(char)
        position = room_bins[char_index % len(room_bins)]

        # Draw circle
        pygame.draw.circle(screen, data["color"], position, circle_radius)

        # Draw character name centered above the circle
        text = font_char.render(char, True, BLACK)
        text_rect = text.get_rect()
        text_rect.center = (position[0], position[1] - text_offset)
        screen.blit(text, text_rect)

    step_text = font_room.render(f"Step {step_num}", True, BLACK)
    step_rect = step_text.get_rect(center=(width // 2, height - 30))
    screen.blit(step_text, step_rect)
    # Save the result
    pygame.image.save(screen, os.path.join(folder, f"step_{step_num:03d}.png"))

def process_sequence(args):
    """
    Generate data for one sequence (sequence_{seq_id}) for a given length.
    Return (seq_folder, question, answer) to write into QA CSV later.
    """
    length, seq_id, base_path = args

    # Initialize local pygame (headless)
    screen = init_pygame()

    # List of rooms
    room_names = ["kitchen", "bathroom", "garden", "office", "bedroom", "hallway"]
    # Create room layout
    rooms = generate_room_layout(room_names)
    
    # Prepare bins for each room
    bins = {}
    for room, rect in rooms.items():
        bin_spacing_x = rect.width // 4
        bin_spacing_y = rect.height // 4
        bin_positions = [
            (rect.x + bin_spacing_x,     rect.y + bin_spacing_y),
            (rect.x + (rect.width - bin_spacing_x), rect.y + bin_spacing_y),
            (rect.x + bin_spacing_x,     rect.y + (rect.height - bin_spacing_y)),
            (rect.x + (rect.width - bin_spacing_x), rect.y + (rect.height - bin_spacing_y)),
            (rect.x + rect.width // 2,   rect.y + rect.height // 2)
        ]
        bins[room] = bin_positions

    # Define colors
    BLUE   = (0, 0, 255)
    RED    = (255, 0, 0)
    GREEN  = (0, 255, 0)
    YELLOW = (255, 255, 0)
    PURPLE = (128, 0, 128)

    # Define characters
    characters = {
        "Sandra":  {"color": BLUE,   "room": "bathroom"},
        "Mary":    {"color": RED,    "room": "garden"},
        "John":    {"color": GREEN,  "room": "hallway"},
        "Daniel":  {"color": YELLOW, "room": "office"},
        "Michael": {"color": PURPLE, "room": "kitchen"},
    }

    # Create folder for this particular sequence
    folder = os.path.join(base_path, f"length_{length}", f"sequence_{seq_id:03d}")
    os.makedirs(folder, exist_ok=True)

    # Create a list of available rooms
    rooms_list = list(rooms.keys())

    for char in characters:
        random.seed(datetime.now().microsecond)
        characters[char]["room"] = random.choice(rooms_list)

    # Randomly choose the character for the question
    q_char = random.choice(list(characters.keys()))
    a_room = characters[q_char]["room"]  # this will be updated if q_char moves

    # Write the sequence to CSV
    csv_path = os.path.join(base_path, f"length_{length}", f"sequence_{seq_id:03d}.csv")
    with open(csv_path, "w+", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Step", "Character", "Room"])

        # Draw initial state (step = 0)
        draw_environment(screen, rooms, bins, characters, 0, folder)
        for char in characters:
            writer.writerow([0, char, characters[char]["room"]])

        # Generate random movements
        for step in range(1, length):
            char = random.choice(list(characters.keys()))
            current_room = characters[char]["room"]
            # choose a new room different from the current one
            target_room = random.choice(list(set(rooms_list) - {current_room}))
            characters[char]["room"] = target_room
            # update a_room if the moved character is q_char
            if char == q_char:
                a_room = target_room
            draw_environment(screen, rooms, bins, characters, step, folder)
            writer.writerow([step, char, target_room])

    # Form a QA pair
    question = f"Where was {q_char} last time?"
    answer = a_room

    # Quit pygame in this process
    pygame.quit()

    return (f"length_{length:3d}/sequence_{seq_id:3d}", question, answer)

def generate_random_sequences_parallel():
    """
    Launch parallel generation of sequences for multiple lengths.
    """
    base_path = "data"
    os.makedirs(base_path, exist_ok=True)

    # list of lengths
    lengths = [128]
    num_sequences_per_length = 100

    tasks = []
    for length in lengths:
        for seq_id in range(num_sequences_per_length):
            tasks.append((length, seq_id, base_path))

    # Parallel processing
    with Pool() as pool:
        results = pool.map(process_sequence, tasks)

    # After all processes finish, `results` contains (seq_folder, question, answer)
    qa_pairs_path = os.path.join(base_path, "qa_pairs.csv")
    with open(qa_pairs_path, "w+", newline="") as qa_file:
        writer = csv.writer(qa_file)
        writer.writerow(["Seq_id", "Question", "Answer"])
        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    generate_random_sequences_parallel()
    print("Generation complete.")
