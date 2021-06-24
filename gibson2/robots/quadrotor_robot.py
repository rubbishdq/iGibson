from pdb import set_trace
import pdb
import gym
import numpy as np
import pybullet as p

from gibson2.robots.robot_locomotor import LocomotorRobot
from gibson2.utils.utils import quatXYZWFromRotMat, rotateMatrixFromTwoVec
from gibson2.utils.mesh_util import quat2rotmat, xyzw2wxyz, lookat


class Quadrotor(LocomotorRobot):
    """
    Quadrotor robot
    Reference: https://repository.upenn.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1705&context=edissertations
    Uses robot velocity control
    """

    def __init__(self, config):
        self.config = config
        self.auto_navigation = config.get("auto_navigation", False)
        self.action_dim = 3 if self.auto_navigation else 6
        LocomotorRobot.__init__(self,
                                "quadrotor/quadrotor.urdf",
                                action_dim=self.action_dim,
                                velocity_coef=config.get("velocity", 0.5),
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="velocity")
        self.history_positions = []

    def set_up_continuous_action_space(self):
        """
        Set up continuous action space
        """
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.velocity_coef * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        self.action_list = [[self.torque, 0, 0, 0, 0, 0],
                            [-self.torque, 0, 0, 0, 0, 0],
                            [0, self.torque, 0, 0, 0, 0],
                            [0, -self.torque, 0, 0, 0, 0],
                            [0, 0, self.torque, 0, 0, 0],
                            [0, 0, -self.torque, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def apply_action(self, action):
        """
        Apply policy action. Zero gravity.
        """
        self.history_positions.append(np.array(self.get_position()))
        if not self.auto_navigation:
            real_action = self.policy_action_to_robot_action(action)
            p.setGravity(0, 0, 0)
            p.resetBaseVelocity(self.robot_ids[0], real_action[:3], real_action[3:])
        else:
            real_action = np.zeros(6)
            tgt_view_vector = np.array(action) / np.linalg.norm(action)
            # (1) x,y,z: directly velocity control to target position(cur_obs+action)
            real_action[:3] = tgt_view_vector * self.velocity_coef
            # (2) roll,pitch,yaw: maintain roll=pitch=0, adjust yaw to watch target position
            # ------- eye pose
            # cur_pos = self.eyes.get_position()
            # cur_orn = self.eyes.get_orientation()
            # cur_rpy = self.eyes.get_rpy()
            # rotate_mat = rotateMatrixFromTwoVec(cur_view_vector, tgt_view_vector)
            # cur_view_vector = quat2rotmat(xyzw2wxyz(cur_orn))[:3, :3].dot(np.array([1, 0, 0]))
            # assert np.linalg.norm(self.get_rpy()[2] - np.arctan2(cur_view_vector[0], -cur_view_vector[1])) < 0.1
            cur_rpy = np.array(self.get_rpy())
            tgt_rpy = np.zeros(3)
            tgt_rpy[-1] = np.arctan2(tgt_view_vector[0], -tgt_view_vector[1])
            for i in range(3):
                real_action[i + 3] = np.clip(10 * (tgt_rpy[i] - cur_rpy[i]), -self.velocity_coef, self.velocity_coef)
            real_action[3:5] *= 0.1  # smooth control for roll&pitch
            p.setGravity(0, 0, 0)
            p.resetBaseVelocity(self.robot_ids[0], real_action[:3], real_action[3:])

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  # +x
            (ord('s'),): 1,  # -x
            (ord('d'),): 2,  # +y
            (ord('a'),): 3,  # -y
            (ord('z'),): 4,  # +z
            (ord('x'),): 5,  # -z
            (): 6
        }

    def robot_specific_reset(self):
        self.history_positions = []
        return super().robot_specific_reset()
