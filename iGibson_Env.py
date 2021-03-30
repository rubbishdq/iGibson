import numpy as np
from functools import reduce
import gibson2
from gibson2.envs.igibson_env import iGibsonEnv as inner_iGibsonEnv
from collections import defaultdict

class iGibsonEnv(object):

    def __init__(self, args, scene_id):
        self.num_agents = args.num_agents
        self.mode = args.mode
        self.scenario_name = args.scenario_name
        self.config = gibson2.__path__[0] + '/examples/configs/' + str(self.scenario_name) + '.yaml'
        # self.config['scene_id'] = scene_id
        # self.config['load_scene_episode_config'] = True
        # self.config['scene_episode_config_name'] = json_file
        self.env = inner_iGibsonEnv(config_file=self.config,
                                    mode=self.mode,
                                    action_timestep=1.0 / 10.0,
                                    physics_timestep=1.0 / 40.0,
                                    device_idx=args.render_gpu_id)

        self.observation_space = []
        self.share_observation_space = []
        self.action_space = []
        for agent_id in range(self.num_agents):
            self.observation_space.append(self.env.observation_space)
            self.share_observation_space.append(self.env.observation_space)
            self.action_space.append(self.env.action_space)

    def seed(self, seed=None):
        if seed is None:
            self.env.seed(1)
        else:
            self.env.seed(seed)

    def reset(self):
        obs = self.env.reset()
        return obs, obs, None

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        return obs, obs, rewards, dones, infos, None
 
    def close(self):
        self.env.close()
