from gibson2.reward_functions.reward_function_base import BaseRewardFunction
from gibson2.utils.utils import quat_pos_to_mat
import numpy as np


class MapExploreReward(BaseRewardFunction):
    """
    Map explore reward
    Encourage exploration of new place. Return zero if no explore.
    """

    def __init__(self, config):
        super(MapExploreReward, self).__init__(config)
        self.scale = self.config.get('map_explore_reward_scale', 10.0)
        # TODO:

    def get_reward(self, task, env, robot_id=0):
        """
        TODO:
        """
        reward = self.scale * task.increase_ratios[robot_id]
        # print(f"explore reward: {reward:.2f}")
        return reward
