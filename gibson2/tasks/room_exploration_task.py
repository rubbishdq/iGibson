import pdb
import pybullet as p
from gibson2.tasks.task_base import BaseTask
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.termination_conditions.max_collision import MaxCollision
from gibson2.termination_conditions.timeout import Timeout
from gibson2.termination_conditions.out_of_bound import OutOfBound
from gibson2.reward_functions.map_explore_reward import MapExploreReward
from gibson2.reward_functions.collision_reward import CollisionReward

from gibson2.utils.utils import l2_distance, rotate_vector_3d, cartesian_to_polar
from gibson2.utils.utils import quat_pos_to_mat, quatxyzw_pos_to_mat
from gibson2.utils.mesh_util import quat2rotmat, xyzw2wxyz, lookat

import numpy as np
import logging
import threading
from collections import defaultdict, OrderedDict
from sklearn.neighbors import KDTree
from icecream import ic

def intersect_part(seg1, seg2):
    """Get intersect part of two 1-d line segments. Returning two values.

    If doesn't intersect, the first return value will be false. Otherwise it will be true, and the second return value will be the intersect part.
    """
    if seg1[0] < seg2[0]:
        if seg1[1] < seg2[0]:
            return False, None
        else:
            seg_inter = [seg2[0], min(seg1[1], seg2[1])]
            return True, seg_inter
    else:
        if seg2[1] < seg1[0]:
            return False, None
        else:
            seg_inter = [seg1[0], min(seg2[1], seg1[1])]
            return True, seg_inter

class LocalGrid:
    def __init__(self):
        self.grid_array = np.array([[[]]], dtype=np.uint8) # 0(00) for unknown, 2(10) for occupied, 3(11) for free
        self.lower_lefts = np.array([], dtype=np.int32)
        self.grid_size = np.array([], dtype=np.int32)
        self.lock = threading.Lock()

    def read_data(self):
        self.lock.acquire()
        grid_array = self.grid_array.copy()
        lower_lefts = self.lower_lefts.copy()
        grid_size = self.grid_size.copy()
        self.lock.release()
        return grid_array, lower_lefts, grid_size
    
    def write_data(self, grid_array, lower_lefts, grid_size, copy=False):
        self.lock.acquire()
        if copy:
            self.grid_array = grid_array.copy()
            self.lower_lefts = lower_lefts.copy()
            self.grid_size = grid_size.copy()
        else:
            self.grid_array = grid_array
            self.lower_lefts = lower_lefts
            self.grid_size = grid_size
        self.lock.release()
    

class GlobalGrid:
    def __init__(self, config):
        self.resolution = config['global_grid']['grid_resolution']
        self.grid_size = np.array(config['global_grid']['grid_size'], dtype=np.int32)
        for s in self.grid_size:
            if s%2 == 1:
                print('Error: at least one element of grid_size in global_grid is not even')
                exit()
        self.grid_array = np.zeros(shape=self.grid_size, dtype=np.uint8)
        pass

    def merge_local_grid(self, local_grid):
        """Merge local voxel grid data into global voxel grid
        
        Data in local_grid may be 0, 2 or 3, representing unknown, occupied or free voxels. 

        Args:
            local_grid (LocalGrid)
        
        merging process:

            old     00      10      11
        new
        00          00      10      11
        10          10      10      10
        11          11      11      11
        
        Use bitwise computation to accelerate merging process.

        R2 = N2 | O2
        R1 = N1 | ((!N2) & O1)
        """
        local_grid_array, local_lower_lefts, local_grid_size = local_grid.read_data()
        if len(local_lower_lefts) != 3 or len(local_grid_size) != 3:
            return
        local_upper_rights = local_lower_lefts + local_grid_size - 1
        global_lower_lefts = -self.grid_size // 2
        global_upper_rights = self.grid_size // 2 - 1

        inter_lower_lefts = np.zeros(3, dtype=np.int32)
        inter_upper_rights = np.zeros(3, dtype=np.int32)
        for i in range(3):
            inter, inter_part = intersect_part([local_lower_lefts[i], local_upper_rights[i]], \
                                                [global_lower_lefts[i], global_upper_rights[i]])
            if not inter:  # no need to merge, local grid out of range
                return
            inter_lower_lefts[i] = inter_part[0]
            inter_upper_rights[i] = inter_part[1]
        # N
        local_part = local_grid_array[inter_lower_lefts[0]-local_lower_lefts[0]:inter_upper_rights[0]-local_lower_lefts[0]+1, \
                                    inter_lower_lefts[1]-local_lower_lefts[1]:inter_upper_rights[1]-local_lower_lefts[1]+1, \
                                    inter_lower_lefts[2]-local_lower_lefts[2]:inter_upper_rights[2]-local_lower_lefts[2]+1]
        # O
        global_part = self.grid_array[inter_lower_lefts[0]-global_lower_lefts[0]:inter_upper_rights[0]-global_lower_lefts[0]+1, \
                                    inter_lower_lefts[1]-global_lower_lefts[1]:inter_upper_rights[1]-global_lower_lefts[1]+1, \
                                    inter_lower_lefts[2]-global_lower_lefts[2]:inter_upper_rights[2]-global_lower_lefts[2]+1]
        
        # bitwise computation
        N2 = local_part & 0x02
        N1 = local_part & 0x01
        O2 = global_part & 0x02
        O1 = global_part & 0x01
        N2_not_lower = (np.invert(N2) & 0x02) >> 1
        R2 = N2 | O2
        R1 = N1 | (N2_not_lower & O1)
        updated_part = R2 | R1

        self.grid_array[inter_lower_lefts[0]-global_lower_lefts[0]:inter_upper_rights[0]-global_lower_lefts[0]+1, \
                        inter_lower_lefts[1]-global_lower_lefts[1]:inter_upper_rights[1]-global_lower_lefts[1]+1, \
                        inter_lower_lefts[2]-global_lower_lefts[2]:inter_upper_rights[2]-global_lower_lefts[2]+1] = updated_part


# For simplicity, we set the global map's coordinate same as the env's coordinate.
# i.e. using env.robots[0].get_position() would give you the position in global map coord.

class VoxelGrid():
    def __init__(self, config):
        self.bbox = np.array(config.get('bounding_box', [[-1,1],[-1,1],[-1,1]]))
        self.voxel_grid_size = np.array(config.get('voxel_grid_size', [20, 20, 20]))
        self.N = int(config.get('n_max_voxels', 5000))
        self.T = int(config.get('voxel_inside_points', 35))
        assert self.bbox.shape == (3, 2) and self.voxel_grid_size.shape == (3,)
        self.W, self.H, self.D = self.voxel_grid_size
        self.voxel_size = (self.bbox[:, 1] - self.bbox[:, 0]) / self.voxel_grid_size
        self._voxel_coords = []
        self._voxel_points = OrderedDict()
        self._voxel_features = OrderedDict()
        self._sorted_voxel_idx = []

    @property
    def voxel_coords(self):
        """Voxel coordinates (N, 3) in voxel grid representation.

        NOTE: in order of (D, H, W) 
        """
        self.n_voxels = len(self._voxel_coords)
        if self.n_voxels > self.N:
            return np.array(self._voxel_coords)[self._sorted_voxel_idx][:self.N]
        else:
            empty_cells = np.zeros((self.N - self.n_voxels, 3))
            return np.concatenate([self._voxel_coords, empty_cells], axis=0)

    @property
    def voxel_features(self):
        """Voxel features (N, T, 6) in voxel grid representation."""
        self.n_voxels = len(self._voxel_coords)
        if self.n_voxels > self.N:
            return np.array(list(self._voxel_features.values()))[self._sorted_voxel_idx][:self.N]
        else:
            empty_cells = np.zeros((self.N - len(self._voxel_coords), self.T, 6))
            return np.concatenate([list(self._voxel_features.values()), empty_cells], axis=0)

    @property
    def voxel_occupancy_num(self):
        return len(self._voxel_coords)

    @property
    def voxel_empty_num(self):
        return self.W * self.H * self.D - len(self._voxel_coords)

    @property
    def voxel_points(self):
        """List of all global points grouped in voxel grid."""
        return list(self._voxel_points.values())

    def update(self, new_points):
        """Update new points in pc into voxel grid

        Args:
            new_points (np.array): (N, 3)
        """
        # group points and get unique voxel coords
        new_voxel_coords = ((new_points - self.bbox[:, 0]) / self.voxel_size).astype(np.int32)
        new_voxel_coords = new_voxel_coords[:, [2, 1, 0]] # convert to (D, H, W)
        new_voxel_coords, inv_ind = np.unique(new_voxel_coords, axis=0, return_inverse=True)
        # merge new points into voxel grid
        for i, voxel_coord in enumerate(new_voxel_coords):
            points = new_points[inv_ind == i]
            voxel_hash = voxel_coord.tobytes()
            if voxel_hash not in self._voxel_features:
                self._voxel_coords.append(voxel_coord)
                self._voxel_points[voxel_hash] = []
                self._voxel_features[voxel_hash] = np.zeros((self.T, 6), dtype=np.float32)
            self._voxel_points[voxel_hash] += points.tolist()
            # random sampling if n_points > self.T
            np.random.shuffle(self._voxel_points[voxel_hash])
            points = np.array(self._voxel_points[voxel_hash][:self.T])
            # augment point features
            self._voxel_features[voxel_hash][:len(points), :] = np.concatenate(\
                [points, points - np.mean(points, axis=0)], axis=1)
        # sort voxel based on inside points
        self._sorted_voxel_idx = sorted(range(len(self._voxel_coords)), \
            key=lambda k: len(self._voxel_points[self._voxel_coords[k].tobytes()]), reverse=True)
        if len(self._voxel_coords) > self.N:
            print("INFO: Map overflow!")

    def reset(self):
        self._voxel_coords = []
        self._voxel_points = OrderedDict()
        self._voxel_features = OrderedDict()


class GlobalMap():
    def __init__(self, config):
        # configs
        self.input_hw = (config.get('image_height', 128), config.get('image_width', 128))
        self.dsample_rate = int(config.get('down_sample_rate', 1))
        self.max_num = int(config.get('n_max_points', 20000))
        self.eps = config.get('duplicate_eps', 0.1)
        self.bbox = np.array(config.get('bounding_box', [[-1,1],[-1,1],[-1,1]]))

        # different representation of global map
        self.map_points = None              # Full point cloud
        self.smap_points = None             # Down-sampled point cloud
        self.voxel_grid = VoxelGrid(config) # Voxel grid (hash storage)
        self.global_grid = GlobalGrid(config) # Voxel grid (hash storage)

    def merge_local_pc(self, pc, pose):
        """Merge the current observed point cloud (in local coord) to the global map (in global coord)

        Update voxel grid.

        Args:
            local_points (np.array): (N, 3)
            pose (np.array): camera pose ([4, 4] [R, T] format matrix)

        Returns:
            (float): increase_ratio, normalized in [0, 1]
        """
        # (1) Remove the [0,0,0] points (invalid) in the point cloud
        mask = ((pc != 0.0).sum(2) > 0)
        pc_flat = pc[mask]
        if len(pc_flat) == 0:
            return 0.0
        # (2) Convert the local points into global points [clipped in BBox]
        lpoints_in_gcoord = self.local_to_global(pc_flat, pose)
        lpoints_in_gcoord = np.clip(lpoints_in_gcoord, self.bbox[:, 0], self.bbox[:, 1])

        # (3) Initialize for beginning
        if self.map_points is None:
            self.map_points = np.copy(lpoints_in_gcoord)
            self.smap_points = np.copy(self.random_sample_pc(lpoints_in_gcoord, self.dsample_rate))
            self.voxel_grid.update(self.map_points)
            increase_ratio = 1.0
        # (4) Get the newly observed points and merge into the global map
        else:
            new_points = self.remove_duplicate(lpoints_in_gcoord)
            snew_points = self.random_sample_pc(new_points, self.dsample_rate)
            self.map_points = np.concatenate([self.map_points, new_points], 0)
            self.smap_points = np.concatenate([self.smap_points, snew_points], 0)
            self.voxel_grid.update(new_points)
            increase_ratio = new_points.shape[0] / (self.input_hw[0] * self.input_hw[1])
        return increase_ratio

    def random_sample_pc(self, points, rate):
        """Down sample of new merged points in gmap.

        Args:
            points (np.arrray): new points
            rate (int): down sample rate

        Returns:
            (np.array): sampled new points
        """
        mask = (np.random.rand(points.shape[0]) < (1.0 / rate))
        sampled_points = points[mask]
        return sampled_points

    def remove_duplicate(self, points, sampled=False):
        """Remove the existed points in global map

        Args:
            points (np.array): local point cloud in global coord.
            sampled (bool, optional): down sample. Defaults to False.

        Returns:
            (np.array): new points to be merged into gmap

        Note:
            Here we only use the distance threshold since we assume perfect depth and pose
        """
        if not sampled:
            kdt = KDTree(self.map_points[:,:3], leaf_size=30, metric='euclidean')
        else:
            kdt = KDTree(self.smap_points[:,:3], leaf_size=30, metric='euclidean')

        dist, ind = kdt.query(points[:,:3], k=1, return_distance=True)
        mask = (dist >= self.eps).squeeze(-1)
        new_points = points[mask]
        return new_points

    def local_to_global(self, points, pose):
        """Transform points from camera coordinates into global coorinates.

        Args:
            local_points (np.array): (N, 3)
            pose (np.array): camera pose ([4, 4] [R, T] format matrix)

        Returns:
            (np.array): global points (N, 3)
        """
        h_points = np.concatenate([points, np.ones_like(points)[:,-1:]], -1)
        gpoints = pose.dot(np.transpose(h_points))[:3,:]
        return np.transpose(gpoints)

    def global_to_local(self, map, tgt_pose):
        """Return global map transformed into local camera coordinate

        Args:
            map (np.array): gmap in global coord.
            tgt_pose (np.array): target frame's camera pose (4x4 RT extrinsics matrix)

        Returns:
            (np.array): gmap in local coord.
        """
        world2cam = np.linalg.inv(tgt_pose)
        gmap_in_lcoord = self.local_to_global(map, world2cam)
        return gmap_in_lcoord

    def get_global_map_pc(self, pose=None):
        """Get global map in representation of point-cloud.

        Args:
            pose (np.array, optional): agent's current pose, if not setting, use `np.eye(4)`. Default is `None`.

        Return:
            (np.array): input_map to PointNet for current robot, shape: [max_num, 3]
        """
        N = self.smap_points.shape[0]
        if N > self.max_num:
            print("INFO: Map overflow!")
            input_map = np.copy(self.smap_points[N-self.max_num:, :])
        else:
            input_map = np.concatenate([self.smap_points, np.tile(self.smap_points[:1,:], (self.max_num-N, 1))], 0)
        ego_pose = np.eye(4) if pose is None else pose
        input_map = self.global_to_local(input_map, ego_pose)
        return input_map

    def reset(self):
        self.map_points = None
        self.smap_points = None
        self.voxel_grid.reset()


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

        self.random_init = self.config.get("random_pos", False)
        # used for non-random setting & failure landing
        self.initial_pos = np.array(self.config.get('initial_pos', [[0, 0, 0]]))
        self.initial_rpy = np.zeros((self.num_robots, 3))
        assert len(self.initial_pos.shape) == 2 and self.initial_pos.shape[0] == self.num_robots, \
            'initial_pos must be consistent with robot num'
        assert len(self.initial_rpy.shape) == 2 and self.initial_rpy.shape[0] == self.num_robots, \
            'initial_pos must be consistent with robot num'

        self.img_h = self.config.get('image_height', 128)
        self.img_w = self.config.get('image_width', 128)
        self.gmap = GlobalMap(self.config)
        self.gmap_format = self.config.get('gmap_format', 'voxel')
        self.increase_ratios = np.zeros(env.num_robots)
        self.floor_num = 0

        self.local_grid = LocalGrid()

    def reset_scene(self, env):
        """
        Reset all scene objects.

        :param env: environment instance
        """
        env.scene.reset_scene_objects()
        self.gmap.reset()

    def sample_initial_pose(self, env):
        """
        Sample robot initial pose

        :param env: environment instance
        :return: initial pose
        """
        _, initial_pos = env.scene.get_random_point(floor=self.floor_num)
        initial_rpy = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        return initial_pos, initial_rpy

    def reset_agent(self, env):
        """
        Reset robot initial pose.

        :param env: environment instance
        """
        if self.random_init:
            max_trials = 100
            # cache pybullet state
            # TODO: p.saveState takes a few seconds, need to speed up
            state_id = p.saveState()
            for _ in range(max_trials):
                reset_success = np.zeros(self.num_robots, dtype=bool)
                initial_pos = np.zeros((self.num_robots, 3))
                initial_rpy = np.zeros((self.num_robots, 3))
                for robot_id in range(self.num_robots):
                    initial_pos[robot_id], initial_rpy[robot_id] = self.sample_initial_pose(env)
                    reset_success[robot_id] = env.test_valid_position(env.robots[robot_id], initial_pos[robot_id], initial_rpy[robot_id])
                p.restoreState(state_id)
                if np.all(reset_success):
                    break
            if not np.all(reset_success):
                logging.warning(f"WARNING: Failed to reset robot without collision, reset to fixed point!")
                initial_pos = self.initial_pos
                initial_rpy = self.initial_rpy
            for robot_id in range(self.num_robots):
                env.land(env.robots[robot_id], initial_pos[robot_id], initial_rpy[robot_id])
                env.robots[robot_id].cur_position = initial_pos[robot_id]
            p.removeState(state_id)
        else:
            for robot_id in range(self.num_robots):
                env.land(env.robots[robot_id], self.initial_pos[robot_id], self.initial_rpy[robot_id])
                env.robots[robot_id].cur_position = self.initial_pos[robot_id]
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)

    def global_to_local(self, env, pos):
        """
        Convert a 3D point in global frame to agent's local frame

        :param env: environment instance
        :param pos: a 3D point in global frame
        :return: the same 3D point in agent's local frame
        """
        return rotate_vector_3d(pos - env.robots[0].get_position(),
                                *env.robots[0].get_rpy())

    def step(self, env):
        """
        Use current state to merge the point cloud into global map, calculate the increase ratio.
        """
        obs = env.get_state()
        for robot_id in range(env.num_robots):
            pc = obs['pc'][robot_id]
            pos = env.robots[robot_id].eyes.get_position()
            quat = env.robots[robot_id].eyes.get_orientation()
            mat = quat2rotmat(xyzw2wxyz(quat))[:3, :3]
            view_direction = mat.dot(np.array([1, 0, 0]))
            pose = np.linalg.inv(lookat(pos, pos + view_direction, [0, 0, 1]))
            increase_ratio = self.gmap.merge_local_pc(pc, pose)
            self.increase_ratios[robot_id] = increase_ratio
        # voxel_density = np.array([len(vp) for vp in self.gmap.voxel_grid.voxel_points])
        # ic(voxel_density.argsort()[::-1][0:5])

        # local grid message from ROS is merged here
        # TODO: add support to multi-agent scenarios
        self.gmap.global_grid.merge_local_grid(self.local_grid)

    def get_reward(self, env, robot_id, collision_links, action, info):
        reward, info = super().get_reward(env, robot_id=robot_id, collision_links=collision_links, action=action, info=info)
        env.episode_reward[robot_id] += reward
        return reward, info

    def get_task_obs(self, env):
        """
        Get task-specific observation.
        Here we use the global map in global coordinate as task

        :param env: environment instance
        :return: input map to PointNet for each robot, [num_robots, max_num, 3]
        """
        task_obs = []
        for robot_id in range(env.num_robots):
            pos = env.robots[robot_id].eyes.get_position()
            quat = env.robots[robot_id].eyes.get_orientation()
            pose = quatxyzw_pos_to_mat(pos, quat)
            local_global_map = self.gmap.get_global_map_pc(pose)
            task_obs.append(local_global_map)
        return np.array(task_obs)
