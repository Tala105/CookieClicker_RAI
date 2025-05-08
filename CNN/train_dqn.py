import os
import json
import time
import tensorflow as tf
import numpy as np
from collections import deque
import threading
from CookieClicker.game import Game
from CookieClicker.building import Building
from CookieClicker.upgrade import Upgrade
from Constants import *
from CNN.agente import Agent
from CNN.utils import *
import matplotlib.pyplot as plt
from queue import Queue, Empty

os.makedirs("CNN/Metadata_saved_files", exist_ok=True)
os.makedirs("CNN/Progress_imgs", exist_ok=True)

plt.switch_backend('Agg')

fps = 16 # CPS estimado na main
batch_size = 256
total_history = []
score_history = []
best_sequence = []
max_total_lock = threading.Lock()
buyables = list(range(2 * NUM_BUILDINGS))
state_size = 3 * NUM_BUILDINGS + 2
action_size = len(buyables)

shared_replay_buffer = deque(maxlen=500000)
replay_lock = threading.Lock()
global_agent_lock = threading.Lock()
plot_queue = Queue()
checkpoint_queue = Queue()

checkpoint_counter = 0
checkpoint_counter_lock = threading.Lock()
checkpoint_event = threading.Event()

plot_counter = 0
plot_lock = threading.Lock()
plot_event = threading.Event()

latest_checkpoint = None
latest_metadata_file = None

tf.compat.v1.disable_eager_execution()

global_graph = tf.Graph()
with global_graph.as_default():
    global_agent = Agent(state_size, action_size, graph=global_graph)
    metadata, epsilon, max_total, all_checkpoints, latest_episode = load_from_checkpoint(global_agent)
    history_file = os.path.join("CNN/Metadata_saved_files", "best_history.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            best_history = json.load(f)
            total_history = best_history.get("totals", [])
            score_history = best_history.get("scores", [])

checkpoint_lock = threading.Lock()

def print_condition(episode):
    if episode == 1:
        return True
    if episode % PRINT_INTERVAL == 0:
        return True
    return False

def fancy_gradient(value: float, text: str, min_value: float = 50000, max_value: float = 150000, reverse_it: bool = False) -> str:
    value = max(min_value, min(max_value, value))
    norm = 0 if max_value == min_value else 100 * (value - min_value) / (max_value - min_value)
    if norm <= 50:
        r = int(255 * (norm / 50))
        g = 255
    else:
        r = 255
        g = int(255 * (1 - (norm - 50) / 50))
    if reverse_it:
        r, g = g, r
    b = 0
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def worker(worker_id, num_episodes):
    os.system('cls' if os.name == 'nt' else 'clear')
    global max_total, best_sequence, total_history, score_history, checkpoint_counter, latest_episode, epsilon

    worker_graph = tf.Graph()
    with worker_graph.as_default():
        local_agent = Agent(state_size, action_size, graph=worker_graph)

    local_agent.epsilon = epsilon
    local_thresehold_counter = 0
    worker_max_total = max_total
    local_agent.load_from_agent(global_agent)
    local_RESET_INTERVAL = INITIAL_RESET_INTERVAL
    while local_RESET_INTERVAL < latest_episode:
        local_RESET_INTERVAL = int(2.5 * local_RESET_INTERVAL)

    for episode in range(latest_episode + 1, num_episodes + 1):
        game = Game((640, 480), fps, local_agent)
        state = game.get_state()
        state = np.reshape(state, [1, state_size])
        cumulative_reward = 0.0
        i = 0

        while True:
            i += 1
            reward = 0.0
            threshold = MAX_TIME_SECONDS/7*(i//(120*fps)+1)
            while True:
                action = local_agent.act(state)
                if action < NUM_BUILDINGS:
                    bought: Building = game.buildings[action]
                    break
                else:
                    bought: Upgrade = game.upgrades[action - NUM_BUILDINGS]
                    if bought.buyable:
                        break
            
            required = bought.cost - game.cookies
            cps = game.cps
            click_power = game.click_power
            cookies_per_second = cps + fps * click_power
            time_needed = required / cookies_per_second if cookies_per_second > 0 else float('inf')

            if time_needed > threshold:
                local_thresehold_counter += 1
                reward = -1
                cumulative_reward = local_agent.gamma * cumulative_reward + reward
                next_state = game.step(0)
                next_state = np.reshape(next_state, [1, state_size])
                done = game.total >= GOAL
                with replay_lock:
                    shared_replay_buffer.append((state, action, reward, next_state, done))
                cumulative_reward = local_agent.gamma * cumulative_reward + reward
                state = next_state
                continue
            
            
            while game.cookies < bought.cost and i < fps * MAX_TIME_SECONDS:
                next_state = game.step(0)
                next_state = np.reshape(next_state, [1, state_size])
                state = next_state
                i += 1
                reward -= 0.001
                if i >= fps * MAX_TIME_SECONDS:
                    break

            next_state = game.step(action + 1)
            next_state = np.reshape(next_state, [1, state_size])
            done = i >= fps*MAX_TIME_SECONDS
            reward = reward_engineering_cookie(state[0], action, next_state[0], done, reward)
            local_agent.replay_buffer.append((state, action, reward, next_state, done))

            with replay_lock:
                shared_replay_buffer.append((state, action, reward, next_state, done))

            cumulative_reward = local_agent.gamma * cumulative_reward + reward
            state = next_state

            if len(local_agent.replay_buffer) > batch_size and i % TRAINING_INTERVAL == 0:
                local_agent.replay(batch_size)

            if done:
                if print_condition(episode):
                    print(f"{bcolors.BRIGHT_BLUE}Worker {worker_id:2}{bcolors.ENDC} | {fancy_gradient(int(game.total), f'Total: {int(game.total):6}/{int(max_total):6}', min_value=0, max_value=max(max_total, 100000), reverse_it=True)} | Episode {episode:4}/{NUM_EPISODES} | " +
                          f"Reward: {cumulative_reward:7.3f} | Epsilon: {local_agent.epsilon:.3f} | Reset in: {local_RESET_INTERVAL - episode:4}", flush=True)
                break
        
        
        local_agent.total_history.append(int(game.total))
        local_agent.score_history.append(cumulative_reward)

        if int(game.total) > worker_max_total:
            worker_max_total = int(game.total)

            with max_total_lock:
                if worker_max_total > max_total:
                    max_total = worker_max_total
                    best_sequence = game.get_action_history()
                    print(f"{bcolors.BLINK}{bcolors.GREEN}New global max_total: {max_total:5} by worker {worker_id:2}! epsilon: {local_agent.epsilon:.3f} | Threshold hit {local_thresehold_counter} times{bcolors.ENDC}")
                    
                    with checkpoint_lock:
                        checkpoint_queue.put((global_agent, episode, max_total, best_sequence))
                        save_history(local_agent.total_history, local_agent.score_history, episode, local_thresehold_counter)
                        checkpoint_event.set()

            with global_agent_lock:
                global_agent.load_from_agent(local_agent)
                global_agent.epsilon = local_agent.epsilon

        if episode % PLOT_INTERVAL == 0 and episode:
            
            with plot_lock:
                print(f"{bcolors.BLUE}Worker {worker_id:2} | Episode {episode:4} plotting{bcolors.ENDC}", flush=True)
                plot_queue.put((worker_id, list(range(1, episode+1)), local_agent.total_history[:episode].copy(), local_agent.score_history[:episode].copy()))
                plot_event.set()


        if episode % (local_RESET_INTERVAL) == 0:
            with replay_lock:
                local_agent.replace_replay_buffer(shared_replay_buffer)
            n_replays = len(local_agent.replay_buffer) // batch_size // 6
            print(f"{bcolors.YELLOW}Worker {worker_id:2} | Episode {episode:4} resetting epsilon, syncing buffer, training {n_replays} times{bcolors.ENDC}", flush=True)
            for _ in range(n_replays):
                local_agent.replay(batch_size)
            local_agent.epsilon_reset()
            local_RESET_INTERVAL = int(2.5*local_RESET_INTERVAL)

    return local_agent.total_history, local_agent.score_history


def plot_progress(episodes, total_history, score_history, worker_id):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.plot(episodes, total_history, label=f"Worker {worker_id}")
    plt.xlabel("Episodes")
    plt.ylabel("Total Cookies")
    plt.title("Total Cookies Generated")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(episodes, score_history, label=f"Worker {worker_id}")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title("Reward Progress")
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.scatter(total_history, score_history, label=f"Worker {worker_id}", alpha=0.6)
    
    coef = np.polyfit(np.log10(total_history), score_history, 1)
    poly1d_fn = np.poly1d(coef)
    x_vals = np.linspace(min(total_history), max(total_history), 100)
    plt.plot(x_vals, poly1d_fn(np.log10(x_vals)), color='red', linestyle='--', label='Regression (log scale)')
    plt.xscale("log")
    
    plt.xlabel("Total Cookies (log scale)")
    plt.ylabel("Cumulative Reward")
    plt.title("Reward vs Total Cookies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"CNN/Progress_imgs/progress_worker_{worker_id}.png")
    plt.close()

def save_checkpoint(agent: Agent, episode: int, max_total: int, best_sequence=[]):
    metadata_path = f"CNN/Metadata_saved_files/checkpoint_iterations_{max_total}_metadata.json"
    with open(metadata_path, 'w') as f:
        metadata = {
            'epsilon': agent.epsilon,
            'episode': episode,
            'max_total': max_total,
            'best_sequence': best_sequence
        }
        json.dump(metadata, f)
    checkpoint_path = f"CNN/Metadata_saved_files/checkpoint_iterations_{max_total}.h5"
    agent.save(checkpoint_path)
    print(f"Saved checkpoint (iterations {max_total}, Epsilon: {agent.epsilon:.4f})")

def main():
    global checkpoint_counter, latest_episode, epsilon

    num_workers = 24
    threads = []

    for i in range(num_workers):
        t = threading.Thread(target=worker, args=(i, NUM_EPISODES))
        threads.append(t)
        t.start()

    active_workers = num_workers
    while active_workers > 0:
        if plot_event.wait(timeout=0.1):
            plot_event.clear()
            with plot_lock:
                while True:
                    try:
                        worker_id, episodes, totals, scores = plot_queue.get_nowait()
                        if episodes is None and totals is None and scores is None:
                            active_workers -= 1
                        else:
                            plot_progress(episodes, totals, scores, worker_id)
                    except Empty:
                        break

        if checkpoint_event.wait(timeout=0.1):
            checkpoint_event.clear()
            with checkpoint_lock:
                while True:
                    try:
                        agent, episode, max_tot, best_sequence = checkpoint_queue.get_nowait()
                        if agent is not None and episode is not None and max_tot is not None:
                            save_checkpoint(agent, episode, max_tot, best_sequence)
                    except Empty:
                        break

    for t in threads:
        t.join()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Exiting all threads.")
        import os
        os._exit(1)