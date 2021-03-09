import numpy as np
from functools import reduce
import gibson2
from gibson2.envs.igibson_env import iGibsonEnv as inner_iGibsonEnv


class iGibsonEnv(object):

    def __init__(self, args):
        self.num_agents = args.num_agents
        self.mode = args.mode
        self.scenario_name = args.scenario_name
        self.config = gibson2.__path__[0] + '/examples/configs/' + str(self.scenario_name) + '.yaml'
        self.env = inner_iGibsonEnv(config_file=self.config,
                                    mode=self.mode,
                                    action_timestep=1.0 / 10.0,
                                    physics_timestep=1.0 / 40.0)
        print(self.env.observation_space)
        print(self.env.action_space)
        ### log ###

        # locobot_point_nav
        # Dict(task_obs:Box(4,), rgb:Box(90, 160, 3), depth:Box(90, 160, 1))
        # Box(2,)

        # fetch_reaching
        # Dict(task_obs:Box(4,), rgb:Box(128, 128, 3), depth:Box(128, 128, 1), scan:Box(220, 1))
        # Box(10,)
        
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
        return obs

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        return obs, rewards, dones, infos
 
    def close(self):
        self.env.close()
