import os
import json
import tensorflow as tf
import numpy as np
from collections import deque
import threading
from time import time
from CookieClicker.game import Game
from Constants import NUM_EPISODES, PLOT_INTERVAL, PRINT_INTERVAL, TRAINING_INTERVAL, GOAL, NUM_BUILDINGS
from CNN.agente import Agent
from CNN.utils import reward_engineering_cookie
import matplotlib.pyplot as plt
from queue import Queue

os.system('clear')
plt.switch_backend('Agg')

batch_size = 128
iteration_history = []
score_history = []
best_sequence = []
min_iteration = float('inf')
min_iteration_lock = threading.Lock()
buyables = list(range(2 * NUM_BUILDINGS))
state_size = 3 * NUM_BUILDINGS + 2
action_size = len(buyables)

shared_replay_buffer = deque(maxlen=100000)
replay_lock = threading.Lock()
global_agent_lock = threading.Lock()
plot_queue = Queue()
checkpoint_queue = Queue()

checkpoint_counter = 0
checkpoint_counter_lock = threading.Lock()
checkpoint_event = threading.Event()

latest_episode = -1
epsilon = 1.0
latest_checkpoint = None
latest_metadata_file = None

tf.compat.v1.disable_eager_execution()

global_graph = tf.Graph()
with global_graph.as_default():
    global_agent = Agent(state_size, action_size, graph=global_graph)

    checkpoint_file = 'checkpoint'
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as file:
            content = file.read().strip().splitlines()
            checkpoint_name = None
            all_checkpoints = None

            for line in content:
                if line.startswith('model_checkpoint_path:'):
                    checkpoint_name = line.split(':')[1].strip().strip('"')
                    print(f"Found checkpoint: {checkpoint_name}")
                elif line.startswith('all_model_checkpoint_paths:'):
                    all_checkpoints = line.split(':')[1].strip().strip('"')

            latest_checkpoint = checkpoint_name

    if latest_checkpoint and os.path.exists(latest_checkpoint+'.index'):
        global_agent.load(latest_checkpoint)
        metadata_file = f"{latest_checkpoint.replace('.h5','_metadata.json')}"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                epsilon = metadata.get('epsilon', 1.0)
                min_iteration = metadata.get('min_iteration', float('inf'))
                latest_episode = metadata.get('episode', -1)
                print(f"Resumed from episode {latest_episode}, epsilon: {epsilon:.4f}")
        else:
            print("Metadata file not found for the checkpoint!")
    else:
        print("No checkpoint found. Starting from scratch.")

def get_wait_threshold(progress_counter):
    base = 60
    step_size = 30
    steps = progress_counter // 15
    return base + steps * step_size

checkpoint_lock = threading.Lock()

def worker(worker_id, num_episodes):
    global min_iteration, best_sequence, iteration_history, score_history, checkpoint_counter, latest_episode, epsilon

    worker_graph = tf.Graph()
    with worker_graph.as_default():
        local_agent = Agent(state_size, action_size, graph=worker_graph)

    local_agent.epsilon = epsilon
    worker_min_iteration = min_iteration
    worker_iteration_history = []
    worker_score_history = []

    print(f"Worker {worker_id} starting from episode {latest_episode + 1}")

    for episode in range(latest_episode + 1, num_episodes + 1):
        game = Game((640, 480), 60, local_agent)
        state = game.get_state()
        state = np.reshape(state, [1, state_size])
        progress_counter = 0
        episode_buildings = 0
        threshold = get_wait_threshold(0)
        cumulative_reward = 0.0
        start_time = time()
        i = 0

        while True:
            i += 1
            reward = 1

            while True:
                action = local_agent.act(state)
                if action < NUM_BUILDINGS:
                    bought = game.buildings[action]
                    break
                else:
                    bought = game.upgrades[action - NUM_BUILDINGS]
                    if bought.buyable:
                        break

            current_cookies = game.cookies
            cost = bought.cost
            progress_counter = max(progress_counter, game.total // 1000)

            if action < NUM_BUILDINGS:
                episode_buildings += 1
                threshold = get_wait_threshold(episode_buildings)

            if current_cookies < cost:
                required = cost - current_cookies
                cps = game.cps
                click_power = game.click_power
                cookies_per_second = cps + 60 * click_power
                time_needed = required / cookies_per_second if cookies_per_second > 0 else float('inf')

                if time_needed > threshold:
                    reward = -5
                    next_state = game.step(0)
                    next_state = np.reshape(next_state, [1, state_size])
                    done = game.total >= GOAL
                    with replay_lock:
                        shared_replay_buffer.append((state, action, reward, next_state, done))
                    cumulative_reward = local_agent.gamma * cumulative_reward + reward
                    state = next_state
                    continue

            while game.cookies < bought.cost and game.total < GOAL:
                next_state = game.step(0)
                next_state = np.reshape(next_state, [1, state_size])
                state = next_state
                i += 1
                cumulative_reward -= 0.001

            next_state = game.step(action + 1)
            done = game.total >= GOAL
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward_engineering_cookie(reward, state[0], action, next_state[0], done)
            local_agent.replay_buffer.append((state, action, reward, next_state, done))

            with replay_lock:
                shared_replay_buffer.append((state, action, reward, next_state, done))

            cumulative_reward = local_agent.gamma * cumulative_reward + reward
            state = next_state

            if len(local_agent.replay_buffer) > batch_size and i % TRAINING_INTERVAL == 0:
                local_agent.replay(batch_size)
            if worker_id == 0 and i % 1000 == 0:
                print(f"Worker {worker_id} | Episode {episode}/{NUM_EPISODES} | Iteration {i} | Epsilon: {local_agent.epsilon} | Reward: {cumulative_reward} | Time: {time() - start_time} | Progress: {game.total} | Min: {min_iteration}", flush=True)

            if done:
                if (episode % PRINT_INTERVAL == 0 and episode > 3) or episode == 1 or worker_id == 0:
                    print(f"Worker {worker_id} | Episode {episode}/{NUM_EPISODES} COMPLETED in {i} iterations! | Min: {min_iteration} | Epsilon: {local_agent.epsilon:.3f}", flush=True)
                break

        worker_iteration_history.append(i)
        worker_score_history.append(cumulative_reward)

        if i < worker_min_iteration:
            worker_min_iteration = i

        with min_iteration_lock:
            if worker_min_iteration < min_iteration:
                min_iteration = worker_min_iteration
                best_sequence = game.get_action_history()
                print(f"New global min_iteration: {min_iteration}")

        if episode % 100 == 0:
            with open(f"checkpoint_episode_{episode}_metadata.json", 'w') as f:
                metadata = {
                    'epsilon': local_agent.epsilon,
                    'episode': episode,
                    'min_iteration': min_iteration
                }
                json.dump(metadata, f)

    return worker_iteration_history, worker_score_history

def plot_progress(episodes, iteration_history, score_history, worker_id):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(episodes, iteration_history, label=f"Worker {worker_id}")
    plt.xlabel("Episodes")
    plt.ylabel("Iterations")
    plt.title("Iteration Progress")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(episodes, score_history, label=f"Worker {worker_id}")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title("Reward Progress")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"progress_worker_{worker_id}.png")
    plt.close()

def save_checkpoint(agent, episode, min_iteration):
    agent.save(f"checkpoint_episode_{episode}.h5")
    metadata = {
        "episode": episode,
        "min_iteration": min_iteration,
        "epsilon": agent.epsilon
    }
    with open(f"checkpoint_episode_{episode}_metadata.json", "w") as f:
        json.dump(metadata, f)
    print(f"Saved checkpoint (Episode {episode}, Epsilon: {agent.epsilon:.4f})")

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
        checkpoint_event.wait()

        while not plot_queue.empty():
            worker_id, episodes, iterations, scores = plot_queue.get()
            if episodes is None and iterations is None and scores is None:
                active_workers -= 1
            else:
                plot_progress(episodes, iterations, scores, worker_id)

        while not checkpoint_queue.empty():
            agent, episode, min_iter = checkpoint_queue.get()
            save_checkpoint(agent, episode, min_iter)

        checkpoint_counter = 0

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
