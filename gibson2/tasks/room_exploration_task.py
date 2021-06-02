from gibson2.tasks.task_base import BaseTask
import pybullet as p
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.termination_conditions.max_collision import MaxCollision
from gibson2.termination_conditions.timeout import Timeout
from gibson2.termination_conditions.out_of_bound import OutOfBound
from gibson2.reward_functions.map_explore_reward import MapExploreReward
from gibson2.reward_functions.collision_reward import CollisionReward

from gibson2.utils.utils import l2_distance, rotate_vector_3d, cartesian_to_polar

import numpy as np


class RoomExplorationTask(BaseTask):
    """
    Room Exploration Task
    The goal is to explore the whole room.
    """

    def __init__(self, env):
        super(RoomExplorationTask, self).__init__(env)
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
        ]
        self.reward_functions = [
            CollisionReward(self.config),
            MapExploreReward(self.config),
        ]

        self.initial_pos = np.array(self.config.get('initial_pos', [0, 0, 0]))
        self.initial_orn = np.array(self.config.get('initial_orn', [0, 0, 0]))

        self.floor_num = 0

    def reset_scene(self, env):
        """
        Reset all scene objects.

        :param env: environment instance
        """
        env.scene.reset_scene_objects()

    def reset_agent(self, env):
        """
        Reset robot initial pose.

        :param env: environment instance
        """
        env.land(env.robots[0], self.initial_pos, self.initial_orn)
        self.path_length = 0.0
        self.robot_pos = self.initial_pos[:2]
        self.geodesic_dist = self.get_geodesic_potential(env)
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)

    def get_termination(self, env, collision_links=[], action=None, info={}):
        """
        FIXME:
        """
        raise NotImplementedError

    def global_to_local(self, env, pos):
        """
        Convert a 3D point in global frame to agent's local frame

        :param env: environment instance
        :param pos: a 3D point in global frame
        :return: the same 3D point in agent's local frame
        """
        return rotate_vector_3d(pos - env.robots[0].get_position(),
                                *env.robots[0].get_rpy())

    def get_reward(self, env, robot_id=0, collision_links=[], action=None, info={}):
        """
        FIXME:
        """
        raise NotImplementedError

    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        raise NotImplementedError

    def get_shortest_path(self,
                          env,
                          from_initial_pos=False,
                          entire_path=False):
        """
        Get the shortest path and geodesic distance from the robot or the initial position to the target position

        :param env: environment instance
        :param from_initial_pos: whether source is initial position rather than current position
        :param entire_path: whether to return the entire shortest path
        :return: shortest path and geodesic distance to the target position
        """
        if from_initial_pos:
            source = self.initial_pos[:2]
        else:
            source = env.robots[0].get_position()[:2]
        target = self.target_pos[:2]
        return env.scene.get_shortest_path(
            self.floor_num, source, target, entire_path=entire_path)

    def step_visualization(self, env):
        """
        Step visualization

        :param env: environment instance
        """
        if env.mode != 'gui':
            return

        self.initial_pos_vis_obj.set_position(self.initial_pos)
        self.target_pos_vis_obj.set_position(self.target_pos)

        if env.scene.build_graph:
            shortest_path, _ = self.get_shortest_path(env, entire_path=True)
            floor_height = env.scene.get_floor_height(self.floor_num)
            num_nodes = min(self.num_waypoints_vis, shortest_path.shape[0])
            for i in range(num_nodes):
                self.waypoints_vis[i].set_position(
                    pos=np.array([shortest_path[i][0],
                                  shortest_path[i][1],
                                  floor_height]))
            for i in range(num_nodes, self.num_waypoints_vis):
                self.waypoints_vis[i].set_position(
                    pos=np.array([0.0, 0.0, 100.0]))
