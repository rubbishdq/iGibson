from gibson2.utils.utils import quatToXYZW
from gibson2.envs.env_base import BaseEnv
from gibson2.tasks.room_rearrangement_task import RoomRearrangementTask
from gibson2.tasks.point_nav_fixed_task import PointNavFixedTask
from gibson2.tasks.point_nav_random_task import PointNavRandomTask
from gibson2.tasks.interactive_nav_random_task import InteractiveNavRandomTask
from gibson2.tasks.dynamic_nav_random_task import DynamicNavRandomTask
from gibson2.tasks.reaching_random_task import ReachingRandomTask
from gibson2.sensors.scan_sensor import ScanSensor
from gibson2.sensors.vision_sensor import VisionSensor
from gibson2.robots.robot_base import BaseRobot
from gibson2.external.pybullet_tools.utils import stable_z_on_aabb
from gibson2.sensors.bump_sensor import BumpSensor

from transforms3d.euler import euler2quat
from collections import OrderedDict
import argparse
import gym
import numpy as np
import pybullet as p
import time
import logging
from collections import defaultdict


class iGibsonEnv(BaseEnv):
    """
    iGibson Environment (OpenAI Gym interface)
    """

    def __init__(
        self,
        config_file,
        num_robots=1,
        scene_id=None,
        mode='headless',
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, gui, iggui
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: which GPU to run the simulation and rendering on
        :param render_to_tensor: whether to render directly to pytorch tensors
        :param automatic_reset: whether to automatic reset after an episode finishes
        """
        super(iGibsonEnv, self).__init__(config_file=config_file,
                                         num_robots=num_robots,
                                         scene_id=scene_id,
                                         mode=mode,
                                         action_timestep=action_timestep,
                                         physics_timestep=physics_timestep,
                                         device_idx=device_idx,
                                         render_to_tensor=render_to_tensor)
        
        self.automatic_reset = automatic_reset

    def load_task_setup(self):
        """
        Load task setup
        """
        self.initial_pos_z_offset = self.config.get(
            'initial_pos_z_offset', 0.1)
        # s = 0.5 * G * (t ** 2)
        drop_distance = 0.5 * 9.8 * (self.action_timestep ** 2)
        assert drop_distance < self.initial_pos_z_offset, \
            'initial_pos_z_offset is too small for collision checking'

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(
            self.config.get('collision_ignore_body_b_ids', []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(
            self.config.get('collision_ignore_link_a_ids', []))

        # discount factor
        self.discount_factor = self.config.get('discount_factor', 0.99)

        # domain randomization frequency
        self.texture_randomization_freq = self.config.get(
            'texture_randomization_freq', None)
        self.object_randomization_freq = self.config.get(
            'object_randomization_freq', None)

        # task
        if self.config['task'] == 'point_nav_fixed':
            self.task = PointNavFixedTask(self)
        elif self.config['task'] == 'point_nav_random':
            self.task = PointNavRandomTask(self)
        elif self.config['task'] == 'interactive_nav_random':
            self.task = InteractiveNavRandomTask(self)
        elif self.config['task'] == 'dynamic_nav_random':
            self.task = DynamicNavRandomTask(self)
        elif self.config['task'] == 'reaching_random':
            self.task = ReachingRandomTask(self)
        elif self.config['task'] == 'room_rearrangement':
            self.task = RoomRearrangementTask(self)
        else:
            self.task = None

    def build_obs_space(self, shape, low, high):
        """
        Helper function that builds individual observation spaces
        """
        return gym.spaces.Box(
            low=low,
            high=high,
            shape=shape,
            dtype=np.float32)

    def load_observation_space(self):
        """
        Load observation space
        """
        self.output = self.config['output']
        self.image_width = self.config.get('image_width', 128)
        self.image_height = self.config.get('image_height', 128)
        observation_space = OrderedDict()
        share_observation_space = OrderedDict()
        sensors = OrderedDict()
        vision_modalities = []
        scan_modalities = []

        if 'task_obs' in self.output:
            observation_space['task_obs'] = self.build_obs_space(
                shape=(self.task.task_obs_dim,), low=-np.inf, high=-np.inf)
            share_observation_space['task_obs'] = self.build_obs_space(
                shape=(self.task.task_obs_dim * self.num_robots,), low=-np.inf, high=-np.inf)

        if 'rgb' in self.output:
            observation_space['rgb'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=0.0, high=1.0)
            share_observation_space['rgb'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3 * self.num_robots),
                low=0.0, high=1.0)
            vision_modalities.append('rgb')

        if 'depth' in self.output:
            observation_space['depth'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1),
                low=0.0, high=1.0)
            share_observation_space['depth'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1 * self.num_robots),
                low=0.0, high=1.0)
            vision_modalities.append('depth')
        if 'pc' in self.output:
            observation_space['pc'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=-np.inf, high=np.inf)
            share_observation_space['pc'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3 * self.num_robots),
                low=-np.inf, high=np.inf)
            vision_modalities.append('pc')
        if 'optical_flow' in self.output:
            observation_space['optical_flow'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 2),
                low=-np.inf, high=np.inf)
            share_observation_space['optical_flow'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 2 * self.num_robots),
                low=-np.inf, high=np.inf)
            vision_modalities.append('optical_flow')
        if 'scene_flow' in self.output:
            observation_space['scene_flow'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=-np.inf, high=np.inf)
            share_observation_space['scene_flow'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3 * self.num_robots),
                low=-np.inf, high=np.inf)
            vision_modalities.append('scene_flow')
        if 'normal' in self.output:
            observation_space['normal'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=-np.inf, high=np.inf)
            share_observation_space['normal'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3 * self.num_robots),
                low=-np.inf, high=np.inf)
            vision_modalities.append('normal')
        if 'seg' in self.output:
            observation_space['seg'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1),
                low=0.0, high=1.0)
            share_observation_space['seg'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1 * self.num_robots),
                low=0.0, high=1.0)
            vision_modalities.append('seg')
        if 'rgb_filled' in self.output:  # use filler
            observation_space['rgb_filled'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=0.0, high=1.0)
            share_observation_space['rgb_filled'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3 * self.num_robots),
                low=0.0, high=1.0)
            vision_modalities.append('rgb_filled')
        if 'scan' in self.output:
            self.n_horizontal_rays = self.config.get('n_horizontal_rays', 128)
            self.n_vertical_beams = self.config.get('n_vertical_beams', 1)
            assert self.n_vertical_beams == 1, 'scan can only handle one vertical beam for now'
            observation_space['scan'] = self.build_obs_space(
                shape=(self.n_horizontal_rays * self.n_vertical_beams, 1),
                low=0.0, high=1.0)
            share_observation_space['scan'] = self.build_obs_space(
                shape=(self.n_horizontal_rays * self.n_vertical_beams, 1 * self.num_robots),
                low=0.0, high=1.0)
            scan_modalities.append('scan')
        if 'occupancy_grid' in self.output:
            self.grid_resolution = self.config.get('grid_resolution', 128)
            self.occupancy_grid_space = gym.spaces.Box(low=0.0,
                                                       high=1.0,
                                                       shape=(self.grid_resolution,
                                                              self.grid_resolution, 1))
            observation_space['occupancy_grid'] = self.occupancy_grid_space
            share_observation_space['occupancy_grid'] = gym.spaces.Box(low=0.0,
                                                       high=1.0,
                                                       shape=(self.grid_resolution,
                                                              self.grid_resolution, 1 * self.num_robots))
            scan_modalities.append('occupancy_grid')

        if 'bump' in self.output:
            observation_space['bump'] = gym.spaces.Box(low=0.0,
                                                       high=1.0,
                                                       shape=(1,))
            share_observation_space['bump'] = gym.spaces.Box(low=0.0,
                                                       high=1.0,
                                                       shape=(1 * self.num_robots,))
            sensors['bump'] = BumpSensor(self)

        if len(vision_modalities) > 0:
            sensors['vision'] = VisionSensor(self, vision_modalities)
            self.simulator.sensor = VisionSensor(self, vision_modalities)

        if len(scan_modalities) > 0:
            sensors['scan_occ'] = ScanSensor(self, scan_modalities)

        self.observation_space = gym.spaces.Dict(observation_space)
        self.share_observation_space = gym.spaces.Dict(share_observation_space)
        self.sensors = sensors

    def load_action_space(self, robot_id=0):
        """
        Load action space
        """
        self.action_space = self.robots[robot_id].action_space

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping
        """
        self.current_step = 0
        self.collision_step = np.zeros(self.num_robots)
        self.current_episode = 0
        self.collision_links = [[]] * self.num_robots

    def load(self):
        """
        Load environment
        """
        super(iGibsonEnv, self).load()
        self.load_task_setup()
        self.load_action_space()
        self.load_observation_space()
        self.load_miscellaneous_variables()

    def get_state(self, collision_links=[]):
        """
        Get the current observation

        :param collision_links: collisions from last physics timestep
        :return: observation as a dictionary
        """
        state = defaultdict(list)

        if 'task_obs' in self.output:
            state['task_obs'] = self.task.get_task_obs(self)

        for key in self.sensors.keys():
            if key in ['vision','scan_occ']:
                key_obs = self.sensors[key].get_obs(self)
                for modality in key_obs[0]:
                    for robot_id in range(self.num_robots):
                        state[modality].append(key_obs[robot_id][modality])
            if key in ['bump']:
                state[key] = self.sensors[key].get_obs(self)
        return state

    def run_simulation(self, robot_id):
        """
        Run simulation for one action timestep (same as one render timestep in Simulator class)

        :return: collision_links: collisions from last physics timestep
        """
        self.simulator_step()
        collision_links = list(p.getContactPoints(
            bodyA=self.robots[robot_id].robot_ids[0]))
        return self.filter_collision_links(collision_links, robot_id)

    def filter_collision_links(self, collision_links, robot_id):
        """
        Filter out collisions that should be ignored

        :param collision_links: original collisions, a list of collisions
        :return: filtered collisions
        """
        new_collision_links = []
        for item in collision_links:
            # ignore collision with body b
            if item[2] in self.collision_ignore_body_b_ids:
                continue

            # ignore collision with robot link a
            if item[3] in self.collision_ignore_link_a_ids:
                continue

            # ignore self collision with robot link a (body b is also robot itself)
            if item[2] == self.robots[robot_id].robot_ids[0] and item[4] in self.collision_ignore_link_a_ids:
                continue
            new_collision_links.append(item)
        return new_collision_links

    def populate_info(self, info, robot_id):
        """
        Populate info dictionary with any useful information
        """
        info['episode_length'] = self.current_step
        info['collision_step'] = self.collision_step[robot_id]

    def step(self, actions):
        """
        Apply robot's action.
        Returns the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        self.current_step += 1

        for robot_id, action in enumerate(actions):
            if action is not None:
                self.robots[robot_id].apply_action(action)

            collision_links = self.run_simulation(robot_id)
            self.collision_links[robot_id] = collision_links
            self.collision_step[robot_id] += int(len(collision_links) > 0)

        state = self.get_state()

        dones = []
        rewards = []
        infos = []
        for robot_id in range(self.num_robots):
            info = {}
            reward, info = self.task.get_reward(self, robot_id, self.collision_links[robot_id], actions[robot_id], info)
            done, info = self.task.get_termination(self, robot_id, self.collision_links[robot_id], actions[robot_id], info)
            self.task.step(self)
            self.populate_info(info, robot_id)

            dones.append(done)
            rewards.append([reward])
            infos.append(info)

        # if done and self.automatic_reset:
        #     info['last_observation'] = state
        #     state = self.reset()
        print(rewards)
        return state, rewards, dones, infos

    def check_collision(self, body_id):
        """
        Check with the given body_id has any collision after one simulator step

        :param body_id: pybullet body id
        :return: whether the given body_id has no collision
        """
        self.simulator_step()
        collisions = list(p.getContactPoints(bodyA=body_id))

        if logging.root.level <= logging.DEBUG:  # Only going into this if it is for logging --> efficiency
            for item in collisions:
                logging.debug('bodyA:{}, bodyB:{}, linkA:{}, linkB:{}'.format(
                    item[1], item[2], item[3], item[4]))

        return len(collisions) == 0

    def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
        """
        Reset position and orientation for the robot or the object

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param offset: z offset
        """
        if orn is None:
            orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        if offset is None:
            offset = self.initial_pos_z_offset

        is_robot = isinstance(obj, BaseRobot)
        body_id = obj.robot_ids[0] if is_robot else obj.body_id
        # first set the correct orientation
        obj.set_position_orientation(pos, quatToXYZW(euler2quat(*orn), 'wxyz'))
        # compute stable z based on this orientation
        stable_z = stable_z_on_aabb(body_id, [pos, pos])
        # change the z-value of position with stable_z + additional offset
        # in case the surface is not perfect smooth (has bumps)
        obj.set_position([pos[0], pos[1], stable_z + offset])

    def test_valid_position(self, obj, pos, orn=None):
        """
        Test if the robot or the object can be placed with no collision

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :return: validity
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if is_robot else obj.body_id
        has_collision = self.check_collision(body_id)
        return has_collision

    def land(self, obj, pos, orn):
        """
        Land the robot or the object onto the floor, given a valid position and orientation

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if is_robot else obj.body_id

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()
            if len(p.getContactPoints(bodyA=body_id)) > 0:
                land_success = True
                break

        if not land_success:
            print("WARNING: Failed to land")

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode
        """
        self.current_episode += 1
        self.current_step = 0
        self.collision_step = np.zeros(self.num_robots)
        self.collision_links = [[]] * self.num_robots

    def randomize_domain(self):
        """
        Domain randomization
        Object randomization loads new object models with the same poses
        Texture randomization loads new materials and textures for the same object models
        """
        if self.object_randomization_freq is not None:
            if self.current_episode % self.object_randomization_freq == 0:
                self.reload_model_object_randomization()
        if self.texture_randomization_freq is not None:
            if self.current_episode % self.texture_randomization_freq == 0:
                self.simulator.scene.randomize_texture()

    def reset(self):
        """
        Reset episode
        """
        self.randomize_domain()
        # move robot away from the scene
        for robot_id in range(self.num_robots):
            self.robots[robot_id].set_position([100.0+robot_id, 100.0+robot_id, 100.0+robot_id])
        
        self.task.reset_scene(self)
        self.task.reset_agent(self)
        self.simulator.sync()
        state = self.get_state()
        self.reset_variables()

        return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui', 'iggui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')
    args = parser.parse_args()

    env = iGibsonEnv(config_file=args.config,
                     mode=args.mode,
                     action_timestep=1.0 / 10.0,
                     physics_timestep=1.0 / 40.0)

    step_time_list = []
    for episode in range(100):
        print('Episode: {}'.format(episode))
        start = time.time()
        env.reset()
        for _ in range(100):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            print('reward', reward)
            if done:
                break
        print('Episode finished after {} timesteps, took {} seconds.'.format(
            env.current_step, time.time() - start))
    env.close()
