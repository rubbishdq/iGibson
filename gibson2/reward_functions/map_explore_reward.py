from gibson2.reward_functions.reward_function_base import BaseRewardFunction


class MapExploreReward(BaseRewardFunction):
    """
    Map explore reward
    Encourage exploration of new place. Return zero if no explore.
    """

    def __init__(self, config):
        super(MapExploreReward, self).__init__(config)
        # TODO:

    def get_reward(self, task, env, robot_id=0):
        """
        TODO:
        """
        gmap = task.gmap
        obs = env.get_state()  # local map
        # obs type: {'rgb': np.array, 'depth': np.array}
        pos = env.robot[robot_id].get_position()  # FIXME: Multi-agent support
        rpy = env.robots[robot_id].get_rpy()
        return 0.0
