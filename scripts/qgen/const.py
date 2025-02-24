SEED = 0xBADFACE

WIDTH, HEIGHT = 512, 512

SEQ_LENGTHS = [1, 2, 4, 8, 16, 32, 64, 128]  # [2 ** i for i in range(8)]
N_QUESTIONS = 50

ROOMS: list[str] = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Hallway"]
CHARS: list[str] = ["Sandra", "Mary", "John", "Daniel", "Michael"]
NOBODY: str = "Nobody"

AnswerTypePerson, AnswerTypeRoom, AnswerTypeNumber = "person", "room", "number"

BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
COLORS = [BLUE, RED, GREEN, YELLOW, PURPLE]
