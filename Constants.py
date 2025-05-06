PLOT_INTERVAL = 100
PRINT_INTERVAL = PLOT_INTERVAL
INITIAL_RESET_INTERVAL = 100
TRAINING_INTERVAL = PLOT_INTERVAL/4
NUM_EPISODES = 50*PLOT_INTERVAL

MAX_TIME_SECONDS = 60*7 # 7 minutes
GOAL = 10**6

NAMES = ["Cursor", "Veia", "Fazenda", "Mina", "Fabrica"]
BUILDING_IDS = ["product0", "product1", "product2", "product3", "product4"]
BUILDING_COSTS = [15, 100, 1100, 12000, 130000]
CPS = [0.1, 1, 8, 47, 260]
NUM_BUILDINGS = len(NAMES)

UPGRADES_IDS = {1: [0, 1, 2, 3, 4, 5, 6], 2:[7, 8, 9, 44], 3:[10, 11, 12, 45], 4:[16, 17, 18, 47], 5:[13, 14, 15, 46]}
UPGRADE_COSTS = [100, 1000, 11000, 120000, 1300000]
UPGRADE_COSTS_GROWTH = {1:[5, 20, 10, 10, 10], 2: [5, 10, 10, 10], 3:[5, 10, 10, 10], 4:[5, 10, 10, 10], 5:[5, 10, 10, 10]}
UPGRADE_REQUIREMENTS = {1:[1, 1, 25, 50, 100], 2:[1, 5, 25, 50, 100], 3:[1, 5, 25, 50, 100], 4:[1, 5, 25, 50, 100], 5:[1, 5, 25, 50, 100]}

CHECKPOINT_FILE = 'CNN/Metadata_saved_files/checkpoint'

STATE_SIZE = 6 + NUM_BUILDINGS * 2 + 1
ACTION_SIZE = 2 + NUM_BUILDINGS * 2 + 1

class bcolors:
    # Estilos
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    INVERT = '\033[7m'
    HIDDEN = '\033[8m'
    
    # Reset
    ENDC = '\033[0m'
    RESET = '\033[0m'

    # Cores de texto
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Cores de texto brilhantes
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Cores de fundo
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

    # Cores de fundo brilhantes
    BG_BRIGHT_BLACK = '\033[100m'
    BG_BRIGHT_RED = '\033[101m'
    BG_BRIGHT_GREEN = '\033[102m'
    BG_BRIGHT_YELLOW = '\033[103m'
    BG_BRIGHT_BLUE = '\033[104m'
    BG_BRIGHT_MAGENTA = '\033[105m'
    BG_BRIGHT_CYAN = '\033[106m'
    BG_BRIGHT_WHITE = '\033[107m'