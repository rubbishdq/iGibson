from gibson2.tasks.task_base import BaseTask
import pybullet as p
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.termination_conditions.max_collision import MaxCollision
from gibson2.termination_conditions.timeout import Timeout
from gibson2.termination_conditions.out_of_bound import OutOfBound
from gibson2.reward_functions.potential_reward import PotentialReward

import logging
import numpy as np


class RoomRearrangementTask(BaseTask):
    """
    Room Rearrangement Task
    The goal is to close as many furniture (e.g. cabinets and fridges) as possible
    """

    def __init__(self, env):
        super(RoomRearrangementTask, self).__init__(env)
        assert isinstance(env.scene, InteractiveIndoorScene), \
            'room rearrangement can only be done in InteractiveIndoorScene'
        self.prismatic_joint_reward_scale = self.config.get(
            'prismatic_joint_reward_scale', 1.0)
        self.revolute_joint_reward_scale = self.config.get(
            'revolute_joint_reward_scale', 1.0)
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
        ]
        self.reward_functions = [
            PotentialReward(self.config),
        ]
        self.floor_num = 0

    def get_potential(self, env):
        """
        Compute task-specific potential: furniture joint positions

        :param env: environment instance
        :param: task potential
        """
        task_potential = 0.0
        for (body_id, joint_id) in self.body_joint_pairs:
            j_type = p.getJointInfo(body_id, joint_id)[2]
            j_pos = p.getJointState(body_id, joint_id)[0]
            scale = self.prismatic_joint_reward_scale \
                if j_type == p.JOINT_PRISMATIC \
                else self.revolute_joint_reward_scale
            task_potential += scale * j_pos
        return task_potential

    def reset_scene(self, env):
        """
        Reset all scene objects and then open certain object categories of interest.

        :param env: environment instance
        """
        env.scene.reset_scene_objects()
        env.scene.force_wakeup_scene_objects()
        self.body_joint_pairs = env.scene.open_all_objs_by_categories(
            ['bottom_cabinet',
             'bottom_cabinet_no_top',
             'top_cabinet',
             'dishwasher',
             'fridge',
             'microwave',
             'oven',
             'washer'
             'dryer',
             ], mode='random', prob=0.5)

    def sample_initial_pose(self, env):
        """
        Sample robot initial pose

        :param env: environment instance
        :return: initial pose
        """
        _, initial_pos = env.scene.get_random_point(floor=self.floor_num)
        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        return initial_pos, initial_orn

    def reset_agent(self, env):
        """
        Reset robot initial pose.
        Sample initial pose, check validity, and land it.

        :param env: environment instance
        """
        reset_success = False
        max_trials = 100

        # cache pybullet state
        # TODO: p.saveState takes a few seconds, need to speed up
        state_id = p.saveState()
        for _ in range(max_trials):
            reset_success = np.zeros(self.num_robots, dtype=bool)
            initial_pos = np.zeros((self.num_robots, 3))
            initial_orn = np.zeros((self.num_robots, 3))
            for robot_id in range(self.num_robots):
                initial_pos[robot_id], initial_orn[robot_id] = self.sample_initial_pose(env)
                reset_success[robot_id] = env.test_valid_position(env.robots[robot_id], initial_pos[robot_id], initial_orn[robot_id])
            p.restoreState(state_id)
            if np.all(reset_success):
                break

        if not np.all(reset_success):
            logging.warning("WARNING: Failed to reset robot without collision")

        for robot_id in range(self.num_robots):
            env.land(env.robots[robot_id], initial_pos[robot_id], initial_orn[robot_id])
        p.removeState(state_id)

        for reward_function in self.reward_functions:
            reward_function.reset(self, env)

    def get_task_obs(self, env):
        """
        No task-specific observation
        """
        return
