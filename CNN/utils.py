import json
import os
from Constants import CHECKPOINT_FILE


def reward_engineering_cookie(reward, state, action, next_state, done):
    """
    Makes reward engineering to allow faster training in the Mountain Car environment.

    :param reward: baseline reward.
    :type reward: float.
    :param state: state.
    :type state: NumPy array with dimension (1, 17).
    :param action: action.
    :type action: int.
    :param next_state: next state.
    :type next_state: NumPy array with dimension (1, 17).
    :param done: if the simulation is over after this experience.
    :type done: bool.
    :return: the new reward.
    :rtype: float.
    """
    # Todo: implement reward engineering
    reward += next_state[1] - state[1]
    return reward

def save_history(iteration_history, score_history, episode):
    with open(f"CNN/Metadata_saved_files/best_history.json", 'w') as f:
        history = {
            'iterations': iteration_history,
            'scores': score_history,
            'episode': episode
        }
        json.dump(history, f)

def load_history():
    if not os.path.exists(f"CNN/Metadata_saved_files/best_history.json"):
        return [], [], -1
    with open(f"CNN/Metadata_saved_files/best_history.json", 'r') as f:
        history = json.load(f)
        iterations = history['iterations']
        scores = history['scores']
        episode = history['episode']
    return iterations, scores, episode


def load_from_checkpoint(global_agent):
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as file:
            content = file.read().strip().splitlines()
            checkpoint_name = None
            all_checkpoints = None

            for line in content:
                if line.startswith('model_checkpoint_path:'):
                    checkpoint_name = line.split(':')[1].strip().strip('"')
                elif line.startswith('all_model_checkpoint_paths:'):
                    all_checkpoints = line.split(':')[1].strip().strip('"')

            latest_checkpoint = os.path.join("CNN/Metadata_saved_files", checkpoint_name)

    if latest_checkpoint and os.path.exists(latest_checkpoint + '.index'):
        global_agent.load(latest_checkpoint)
        metadata_file = f"{latest_checkpoint.replace('.h5', '_metadata.json')}"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                epsilon = metadata.get('epsilon', 1.0)
                min_iteration = metadata.get('min_iteration', float('inf'))
                latest_episode = metadata.get('episode', 0)
            global_agent.epsilon = epsilon
            global_agent.load(latest_checkpoint)
            global_agent.iteration_history, global_agent.score_history, latest_episode = load_history()
            print(f"Loaded checkpoint (Episode {latest_episode}, Epsilon: {epsilon})")
    else:
        print("No checkpoint found. Starting from scratch.")

    return metadata, epsilon, min_iteration, all_checkpoints, latest_episode