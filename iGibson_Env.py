import pdb
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
        config_file_name = gibson2.__path__[0] + '/examples/configs/' + str(self.scenario_name) + '.yaml'
        self.env = inner_iGibsonEnv(config_file=config_file_name,
                                    num_robots=self.num_agents,
                                    scene_id=scene_id,
                                    mode=self.mode,
                                    action_timestep=1.0 / 10.0,
                                    physics_timestep=1.0 / 40.0,
                                    device_idx=args.render_gpu_id,
                                    user_args=args)
        self.observation_space = []
        self.share_observation_space = []
        self.action_space = []
        for agent_id in range(self.num_agents):
            self.observation_space.append(self.env.observation_space)
            self.share_observation_space.append(self.env.share_observation_space)
            self.action_space.append(self.env.action_space)

    def seed(self, seed=None):
        if seed is None:
            self.env.seed(1)
        else:
            self.env.seed(seed)

    def reset(self):
        obs = self.env.reset()
        obs, share_obs = self.generate_share_obs(obs)
        return obs, share_obs, None

    def generate_share_obs(self, obs):
        share_obs = {}
        for key in obs.keys():
            if key in ['rgb', 'depth', 'pc']:
                # [agent, height, width, channel]
                obs[key] = np.array(obs[key])
                # [height, width, channel*agent]
                share_obs[key] = np.concatenate(obs[key], axis=np.argmin(obs[key][0].shape))
                # [agent, height, width, channel*agent]
                share_obs[key] = np.expand_dims(share_obs[key], 0).repeat(self.num_agents, axis=0)
            elif key == 'gmap':
                # FIXME: how to define share obs?
                # [agent, n_max_points, 3]
                obs[key] = np.array(obs[key])
                # [n_max_points*agents, 3]
                share_obs[key] = np.concatenate(obs[key], axis=np.argmax(obs[key][0].shape))
                # [agent, n_max_points*agents, 3]
                share_obs[key] = np.expand_dims(share_obs[key], 0).repeat(self.num_agents, axis=0)
            elif key == 'pos':
                # [agent, 3]
                obs[key] = np.array(obs[key])
                # [agent*3]
                share_obs[key] = np.concatenate(obs[key], axis=0)
                # [agent, agent*3]
                share_obs[key] = np.expand_dims(share_obs[key], 0).repeat(self.num_agents, axis=0)
            else:
                raise NotImplementedError

        return obs, share_obs

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        obs, share_obs = self.generate_share_obs(obs)
        return obs, share_obs, rewards, dones, infos, None

    def close(self):
        self.env.close()
