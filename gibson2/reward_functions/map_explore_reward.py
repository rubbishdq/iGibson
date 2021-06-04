from gibson2.reward_functions.reward_function_base import BaseRewardFunction
from gibson2.utils.utils import quat_pos_to_mat

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
        obs = env.get_state()  # local map
        rgb = obs['rgb']
        depth = obs['depth']
        # obs type: {'rgb': np.array, 'depth': np.array}
        pos = env.robot[robot_id].get_position()  # FIXME: Multi-agent support
        quat = env.robots[robot_id].get_orientation()
        pose = quad_pos_to_mat(pos, quat)
        increase_ratio = task.gmap.merge_sampled_local_obs(np.concatenate([rgb, depth], -1), pose)
        reward = self.scale * increase_ratio
        return reward
