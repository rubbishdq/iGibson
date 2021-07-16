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
        self.use_custom_model = config.get("use_custom_model", False)
        model_filename = "quadrotor/quadrotor_custom.urdf" if self.use_custom_model else "quadrotor/quadrotor.urdf"
        self.action_timestep = config.get("action_timestep", 10)
        LocomotorRobot.__init__(self,
                                filename=model_filename,
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
            tgt_view_direction = np.array(action) / np.linalg.norm(action)
            # (1) x,y,z: directly velocity control to target position(cur_obs+action)
            real_action[:3] = tgt_view_direction * self.velocity_coef
            # (2) roll,pitch,yaw: maintain roll=pitch=0, adjust yaw to watch target positionu
            cur_orn = self.eyes.get_orientation()
            cur_view_direction = quat2rotmat(xyzw2wxyz(cur_orn))[:3, :3].dot(np.array([1, 0, 0]))
            delta_yaw = np.arccos(np.clip(np.sum(cur_view_direction[:2] * tgt_view_direction[:2]) / \
                (np.linalg.norm(cur_view_direction[:2]) * np.linalg.norm(tgt_view_direction[:2]) + 1e-8), -1, 1))
            if np.abs(delta_yaw) > self.velocity_coef / self.action_timestep / 2:  # delta_yaw must larger than half of the incoming change
                real_action[5] = np.sign(np.cross(cur_view_direction[:2], tgt_view_direction[:2])) * self.velocity_coef
            real_action[3:5] = -np.array(self.get_rpy())[:2] * self.velocity_coef
            # cur_rpy = np.array(self.get_rpy())
            # tgt_rpy = np.zeros(3)
            # tgt_rpy[-1] = np.arctan2(tgt_view_direction[0], -tgt_view_direction[1])
            # for i in range(3):
            #     real_action[i + 3] = tgt_rpy[i] - cur_rpy[i]
            #     if np.abs(real_action[i + 3]) > np.pi:
            #         real_action[i + 3] -= np.sign(real_action[i + 3]) * 2 * np.pi
            # real_action[-1] *= 3  # fast adjustment for yaw control
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
