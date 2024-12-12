import pygame

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
def draw_environment(step_num):
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
    pygame.image.save(screen, f"../screenshots/step_{step_num:02d}.png")

# Simulate movement
def simulate_movement():
    for step_num, (char, target_room) in enumerate(sequence, start=1):
        characters[char]["room"] = target_room.lower()
        draw_environment(step_num)

if __name__ == "__main__":
    # Movement sequence
    sequence = [
        ("Mary", "hallway"),
        ("Michael", "garden"),
        ("Sandra", "office"),
        ("John", "hallway"),
        ("John", "office"),
        ("Sandra", "hallway"),
        ("Daniel", "office"),
        ("Mary", "office"),
        ("Sandra", "office"),
        ("Michael", "office"),
    ]

    # Extract unique room names from the movement sequence
    room_names = ["kitchen", "bathroom", "garden", "office", "bedroom", "hallway"]

    # Generate room layout dynamically
    rooms = generate_room_layout(list(room_names))
    

    # Initialize bins for each room
    bins = {}

    # Main execution
    draw_environment(0)  # Save initial state
    simulate_movement()

pygame.quit()
