import tensorflow as tf
import gym


class CnnPolicy:
    recurrent = False

    def __init__(self, name, ob_space, ac_space, kind='large'):
        """
        :param name: (str) name of the policy
        :param ob_space:  (gym.spaces.Box) observation space
        :param ac_space:  (gym.spaces.Box) action space
        :param kind:  (str) 'small' or 'large'
        """
        self._init(ob_space, ac_space, kind)
        self.scope = name

    def _init(self, ob_space, ac_space, kind):
        # only support Box observation space
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype =