import pygame
import random
import csv
import os

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1024, 768
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Character Movement Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# Create directories for outputs
os.makedirs("data", exist_ok=True)

# Function to generate room layout
def generate_room_layout(room_names):
    cols = 3  # Number of columns in the grid
    room_width = (WIDTH - (cols + 1) * 50) // cols
    room_height = (HEIGHT - (len(room_names) // cols + 1) * 50) // (len(room_names) // cols)
    layout = {}

    for i, room_name in enumerate(room_names):
        row = i // cols
        col = i % cols
        x = 50 + col * (room_width + 50)
        y = 50 + row * (room_height + 50)
        layout[room_name] = pygame.Rect(x, y, room_width, room_height)

    return layout

# Characters represented as circles
characters = {
    "Sandra": {"color": BLUE, "room": "bathroom"},
    "Mary": {"color": RED, "room": "garden"},
    "John": {"color": GREEN, "room": "hallway"},
    "Daniel": {"color": YELLOW, "room": "office"},
    "Michael": {"color": PURPLE, "room": "kitchen"},
}

# Function to draw the environment
def draw_environment(step_num, folder):
    screen.fill(WHITE)
    for room, rect in rooms.items():
        pygame.draw.rect(screen, BLACK, rect, 2)
        font = pygame.font.Font(None, 24)
        text = font.render(room.capitalize(), True, BLACK)
        screen.blit(text, (rect.x + 10, rect.y + 10))

        # Draw bins for each room (positions for up to 5 people)
        bin_spacing_x = rect.width // 3
        bin_spacing_y = rect.height // 3
        bin_positions = [
            (rect.x + bin_spacing_x, rect.y + bin_spacing_y),
            (rect.x + 2 * bin_spacing_x, rect.y + bin_spacing_y),
            (rect.x + bin_spacing_x, rect.y + 2 * bin_spacing_y),
            (rect.x + 2 * bin_spacing_x, rect.y + 2 * bin_spacing_y),
            (rect.x + rect.width // 2, rect.y + rect.height // 2)
        ]
        bins[room] = bin_positions

    # Draw characters in their current rooms
    for char, data in characters.items():
        room_bins = bins[data["room"].lower()]
        char_index = list(characters.keys()).index(char)
        position = room_bins[char_index % len(room_bins)]
        pygame.draw.circle(screen, data["color"], position, 20)
        font = pygame.font.Font(None, 20)
        text = font.render(char, True, BLACK)
        screen.blit(text, (position[0] - 25, position[1] - 30))

    # Save the current frame to a PNG
    pygame.image.save(screen, f"{folder}/step_{step_num:02d}.png")

# Generate random sequences and QA pairs
def generate_random_sequences():
    rooms_list = list(rooms.keys())
    qa_data = []
    for length in [2, 5, 10, 20, 25]:
        for seq_id in range(100):
            folder = f"data/length_{length}/sequence_{seq_id}"
            os.makedirs(folder, exist_ok=True)
            
            for char in characters:
                characters[char]["room"] = random.choice(rooms_list)

            sequence = [(0, char, info["room"]) for char, info in characters.items()]

            q_char = random.choice(list(characters.keys()))
            a_room = characters[q_char]["room"]

            for step in range(length):
                char = random.choice(list(characters.keys()))
                target_room = random.choice(list(set(rooms_list) - {characters[char]["room"]}))
                a_room = target_room if char == q_char else a_room
                sequence.append((step + 1, char, target_room))

            with open(f"data/length_{length}/sequence_{seq_id}.csv", "w+", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Step", "Character", "Room"])
                draw_environment(0, folder)
                for step_num, char, target_room in sequence:
                    characters[char]["room"] = target_room.lower()
                    if step_num > 0:
                        draw_environment(step_num, folder)
                    writer.writerow([step_num, char, target_room])

            # Generate a QA pair
            question = f"Where was {q_char} last time?"
            answer = f"{a_room}"
            qa_data.append((f"length_{length}/sequence_{seq_id}", question, answer))

    # Save QA pairs to a CSV file
    with open("data/qa_pairs.csv", "w+", newline="") as qa_file:
        writer = csv.writer(qa_file)
        writer.writerow(["Seq_id", "Question", "Answer"])
        writer.writerows(qa_data)

if __name__ == "__main__":
    # Extract unique room names from the movement sequence
    room_names = ["kitchen", "bathroom", "garden", "office", "bedroom", "hallway"]

    # Generate room layout dynamically
    rooms = generate_room_layout(list(room_names))

    # Initialize bins for each room
    bins = {}

    # Generate random sequences and QA pairs
    generate_random_sequences()

pygame.quit()