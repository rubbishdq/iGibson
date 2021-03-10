from gibson2.sensors.sensor_base import BaseSensor
from gibson2.sensors.dropout_sensor_noise import DropoutSensorNoise

import numpy as np
import os
import gibson2
from collections import OrderedDict


class BumpSensor(BaseSensor):
    """
    Bump sensor
    """

    def __init__(self, env):
        super(BumpSensor, self).__init__(env)

    def get_obs(self, env):
        """
        Get Bump sensor reading

        :return: Bump sensor reading
        """
        has_collision = [float(len(env.collision_links[robot_id]) > 0) for robot_id in range(self.num_robots)]
        return has_collision
