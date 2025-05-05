PLOT_INTERVAL = 100
PRINT_INTERVAL = PLOT_INTERVAL
INITIAL_RESET_INTERVAL = 100
TRAINING_INTERVAL = PLOT_INTERVAL/4
NUM_EPISODES = 50*PLOT_INTERVAL
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
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'