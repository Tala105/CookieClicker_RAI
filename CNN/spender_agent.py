import numpy as np

class SpenderAgent:
    """
    Represents a Deep Q-Networks (DQN) agent.
    """
    def __init__(self, state_size, action_size, buy_size):
        """
        Creates a Deep Q-Networks (DQN) agent.

        :param state_size: number of dimensions of the feature vector of the state.
        :type state_size: int.
        :param action_size: number of actions.
        :type action_size: int.
        :param buy_size: number of buyable actions.
        :type buy_size: int.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.buy_size = buy_size

    def act(self, state):
        """
        Chooses an action using an epsilon-greedy policy.

        :param state: current state.
        :type state: NumPy array with dimension (1, 4*NUM_BUILDINGS+3).
        :return: chosen action.
        :rtype: int.
        """
        costs = []
        for i in range(4, len(state[0])):
            if not i % 3 == 1:
                costs.append(state[0][i])
        action = np.argmin(costs)
        return action
