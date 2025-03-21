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
