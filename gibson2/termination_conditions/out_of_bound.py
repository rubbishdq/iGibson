from gibson2.termination_conditions.termination_condition_base import BaseTerminationCondition
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
import numpy as np


class OutOfBound(BaseTerminationCondition):
    """
    OutOfBound used for navigation tasks in InteractiveIndoorScene
    Episode terminates if the robot goes outside the valid region
    """

    def __init__(self, config):
        super(OutOfBound, self).__init__(config)
        self.fall_off_thresh = self.config.get(
            'fall_off_thresh', 0.03)
        self.bounding_box = config.get('bounding_box', None)
        if self.bounding_box is not None:
            self.bounding_box = np.array(self.bounding_box)
            assert self.bounding_box.shape == (3, 2)

    def get_termination(self, task, env, robot_id=0):
        """
        Return whether the episode should terminate.
        Terminate if the robot goes outside the valid region

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = False
        # fall off the cliff of valid region
        if isinstance(env.scene, InteractiveIndoorScene):
            robot_z = env.robots[robot_id].get_position()[2]
            if robot_z < (env.scene.get_floor_height() - self.fall_off_thresh):
                done = True
        # out of the bounding box
        # if self.bounding_box is not None:
        #     cur_pos = np.array(env.robots[robot_id].get_position())
        #     done = np.any([cur_pos < self.bounding_box[:, 0], cur_pos > self.bounding_box[:, 1]]) or done
        success = False
        return done, success
