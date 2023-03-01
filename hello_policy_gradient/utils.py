import numpy as np

from config import gamma


def discount_rewards(reward_list):
    """
    Calculate the discounted rewards

    :param reward_list: (list) reward_list contains all rewards in one episode(1 training episode = 21 game episodes(pong specific)
    :return: (list) discounted reward list, shape the same with reward_list
    """
    discounted_r = np.zeros_like(reward_list)
    running_add = 0
    for t in reversed(range(0, reward_list.size)):
        # pong only has non-zero reward +1 or -1 when a game ends
        # reset the sum, since this was a game boundary (pong specific!)
        if reward_list[t] != 0:
            running_add = 0
        running_add = running_add * gamma + reward_list[t]
        discounted_r[t] = running_add
    return discounted_r
