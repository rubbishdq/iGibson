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

import cv2
from sklearn.neighbors import NearestNeighbors

# For simplicity, we set the global map's coordinate same as the env's coordinate.
# i.e. using env.robots[0].get_position() would give you the position in global map coord.

class globalMap():
    def __init__(self, K, input_hw, dsample_rate=1.0):
        # K: camera intrinsics
        # init_pos: The starting position (in env coordinate) of the current scene
        # init_pose: The starting camera pose (extrinsics)
        # input_hw: The height and width of input rgbd image
        # dsample_rate: Down-sample rate to store the map (point cloud)
        self.input_hw = input_hw
        self.dsample_rate = int(dsample_rate)

        self.K = np.array(K)
        self.K_inv = np.linalg.inv(K)
        self.hxy = self.gen_meshgrid(self.input_hw, 1.0) # [3, h*w]

        self.sK = self.scale_K(self.K, self.dsample_rate)
        self.sK_inv = np.linalg.inv(self.sK)
        self.shxy = self.gen_meshgrid(self.input_hw, self.dsample_rate) # [3, h*w/s]

        self.map_points = None
        self.sampled_map_points = None
    
    def scale_K(self, K, drate):
        # scale the intrinsics to meet the resized image
        scaled_K = np.copy(K)
        scaled_K[0,:] /= drate
        scaled_K[1,:] /= drate
        return scaled_K
    
    def gen_meshgrid(self, hw, rate):
        h = hw[0] / rate
        w = hw[1] / rate
        xx, yy = np.meshgrid([np.range(w), np.range(h)])
        hxy = np.concatenate([xx, yy, np.ones((h,w))])
        return np.transpose(np.reshape(hxy, (-1,3)))

    def unproject(self, rgbd, hxy, K_inv):
        # unproject the colored depth to 3d points, rgbd: [H, W, 4]
        # return: colored points: [N, 6]
        depth = rgbd[...,-1:]
        assert (depth.shape[0]*depth.shape[1] == hxy.shape[1])
        points = K_inv.dot(hxy) * np.reshape(depth, -1)
        rgb = np.reshape(rgbd[...,:3], (-1,3))
        return np.concatenate([np.transpose(points), rgb], -1)
    
    def local_to_global(self, points, pose):
        # Transform the camera coord points into world (env) coord.
        # points: [N, 3], pose: [4, 4] [R, T] format
        h_points = np.concatenate([points, np.ones_like(points)[:,-1:]], -1)
        gpoints = pose.dot(np.transpose(h_points))
        return np.transpose(gpoints)
    
    def remove_duplicate(self, points, sampled=False, eps=1e-3):
        # remove the existed points in global map
        # Note: here we only use the distance threshold since we assume perfect depth and pose
        if not sampled:
            kdt = KDTree(self.map_points[:,:3], leaf_size=30, metric='euclidean')
        else:
            kdt = KDTree(self.sampled_map_points[:,:3], leaf_size=30, metric='euclidean')
        
        dist, ind = kdt.query(points[:,:3], k=1, return_distance=True)
        mask = (dist < eps)
        new_points = points[mask]
        return new_points

    def merge_local_obs(self, rgbd, cur_pose):
        # rgbd observation and agent's pose wrt global (env) coordinate.
        # we provide both original and sampled global maps
        assert rgbd.shape[0] == self.input_hw[0]
        resized_rgbd = cv2.resize(rgbd, (self.input_hw[0] // self.dsample_rate, self.input_hw[1] // self.dsample_rate))
        
        local_points = self.unproject(rgbd, self.K_inv)
        slocal_points = self.unproject(resized_rgbd, self.sK_inv)
        
        lpoints_in_gcoord = self.local_to_global(local_points[:,:3], cur_pose)
        lpoints_in_gcoord = np.concatenate([lpoints_in_gcoord, local_points[:,3:]], -1)
        slpoints_in_gcoord = self.local_to_global(slocal_points[:,:3], cur_pose)
        slpoints_in_gcoord = np.concatenate([slpoints_in_gcoord, slocal_points[:,3:]], -1)
        
        if not (self.map_points is None):
            new_points = self.remove_duplicate(lpoints_in_gcoord)
            self.map_points = np.concatenate([self.map_points, new_points], 0)
        else:
            self.map_points = np.copy(lpoints_in_gcoord)
        
        if not (self.smap_points is None):
            snew_points = self.remove_duplicate(slpoints_in_gcoord)
            self.smap_points = np.concatenate([self.smap_points, snew_points], 0)
        else:
            self.smap_points = np.copy(slpoints_in_gcoord)
    
    def merge_sampled_local_obs(self, rgbd, cur_pose):
        # rgbd observation and agent's pose wrt global (env) coordinate.
        # we provide only sampled global maps using this func
        assert rgbd.shape[0] == self.input_hw[0]
        resized_rgbd = cv2.resize(rgbd, (self.input_hw[0] // self.dsample_rate, self.input_hw[1] // self.dsample_rate))
        
        slocal_points = self.unproject(resized_rgbd, self.sK_inv)
        slpoints_in_gcoord = self.local_to_global(slocal_points[:,:3], cur_pose)
        slpoints_in_gcoord = np.concatenate([slpoints_in_gcoord, slocal_points[:,3:]], -1)
        
        if not (self.smap_points is None):
            snew_points = self.remove_duplicate(slpoints_in_gcoord)
            self.smap_points = np.concatenate([self.smap_points, snew_points], 0)
        else:
            self.smap_points = np.copy(slpoints_in_gcoord)
        
    def global_to_local(self, tgt_pose, sampled=True):
        # return transformed global map into local frame coordinate
        # tgt_pose: target frame's camera pose (4x4 RT extrinsics matrix)
        world2cam = np.linalg.inv(tgt_pose)
        if not sampled:
            gmap_in_lcoord = self.local_to_global(self.map_points, world2cam)
        else:
            gmap_in_lcoord = self.local_to_global(self.smap_points, world2cam)
        return gmap_in_lcoord


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
        self.img_h = self.config.get('image_height', 128)
        self.img_w = self.config.get('image_width', 128)
        self.vfov = self.config.get('vertical_fov', 90)
        fx = (self.img_w / 2.0) / np.tan(self.vfov / 360.0 * np.pi)
        K = np.array([[fx, 0.0, self.img_w/2.0], [0.0, fx, self.img_h/2.0], [0,0,1.0]])
        self.gmap = gMap(K, (self.img_h, self.img_w), 8)

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
