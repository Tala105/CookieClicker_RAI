import random


class RandomAgent:
    """
    Represents a Deep Q-Networks (DQN) agent.
    """
    def __init__(self, buy_size):
        """
        Creates a Deep Q-Networks (DQN) agent.

        :param buy_size: number of buyable actions.
        :type buy_size: int.
        """
        self.buy_size = buy_size

    def act(self, state):
        """
        Chooses an action using an epsilon-greedy policy.

        :param state: current state.
        :type state: NumPy array with dimension (1, 4*NUM_BUILDINGS+3).
        :return: chosen action.
        :rtype: int.
        """
        action = random.randint(0, self.buy_size)
        return action
