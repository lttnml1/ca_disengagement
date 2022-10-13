# Simple Agent

import carla
from enum import Enum
from shapely.geometry import Polygon
import numpy as np
import math
from collections import deque

from ca_disengagement.agents.local_planner import LocalPlanner
from ca_disengagement.agents.global_route_planner import GlobalRoutePlanner
from ca_disengagement.agents.misc import get_speed


class SimpleAgent(object):
    
    def __init__(self, vehicle, end, target_speed=20):
        """
        Initialization the agent paramters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param target_speed: speed (in Km/h) at which the vehicle will move
        """
        self._end = end
        self._k_p = 1.0
        self._k_i = 0.0
        self._k_d = 0.0
        self._dt = 0.03

        self._max_throt = 1.0
        self._max_brake = 0.3

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        
        self._target_speed = target_speed

    def add_emergency_stop(self, control):
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control
    
    def run_step(self):
        """Execute one step of navigation."""
        #vehicle_speed = get_speed(self._vehicle) / 3.6

        throttle, brake = self.pedal(self._target_speed)        
        waypoint = self._end
        steer = self.steer(waypoint, self._vehicle.get_transform())
        ctrl = carla.VehicleControl(throttle = throttle, steer = steer, brake = brake)
        return ctrl

    def set_target_speed(self, speed):
        self._target_speed = speed

    def set_destination(self, end_location):
        self._end = end_location

    def pedal(self,target_speed):
        self._error_buffer = deque(maxlen=10)
        current_speed = get_speed(self._vehicle)
        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        acceleration = np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

        if acceleration >= 0.0:
            throttle = min(acceleration, self._max_throt)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(acceleration), self._max_brake)
        
        return (throttle, brake)

    def steer(self, waypoint, vehicle_transform):
        
        e_buffer = deque(maxlen=10)
        """
        Estimate the steering angle of the vehicle based on the PID equations

            :param waypoint: target waypoint
            :param vehicle_transform: current transform of the vehicle
            :return: steering control in the range [-1, 1]
        """
        # Get the ego's location and forward vector
        ego_loc = vehicle_transform.location
        v_vec = vehicle_transform.get_forward_vector()
        v_vec = np.array([v_vec.x, v_vec.y, 0.0])

        w_loc = waypoint

        w_vec = np.array([w_loc.x - ego_loc.x,
                          w_loc.y - ego_loc.y,
                          0.0])

        wv_linalg = np.linalg.norm(w_vec) * np.linalg.norm(v_vec)
        if wv_linalg == 0:
            _dot = 1
        else:
            _dot = math.acos(np.clip(np.dot(w_vec, v_vec) / (wv_linalg), -1.0, 1.0))
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        e_buffer.append(_dot)
        if len(e_buffer) >= 2:
            _de = (e_buffer[-1] - e_buffer[-2]) / self._dt
            _ie = sum(e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

    def done(self):
        """Check whether the agent has reached its destination."""
        diff_x = self._vehicle.get_transform().location.x - self._end.x
        diff_y = self._vehicle.get_transform().location.y - self._end.y
        return math.sqrt(diff_x**2 + diff_y**2) < 1