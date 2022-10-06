#!/usr/bin/env python

"""
    This is the Base class from which all scenarios are derived
"""

import carla

class Scenario(object):
    def __init__(self, host=None, port=None):
        self.host = host
        self.port = port

    def world_setup(self):
        try:
            self.client = carla.Client(self.host,self.port)
            self.client.set_timeout(4.0)
            self.world = self.client.get_world()
        except RuntimeError:
            print(f"Could not connect to client: {self.host}:{self.port}")
            return 1
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.blueprints = self.world.get_blueprint_library()
            


