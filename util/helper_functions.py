#!/usr/bin/env python

import carla

def draw_spawn_point_locations(world, spawn_points):
    for i in range(len(spawn_points)):
        world.debug.draw_string(spawn_points[i].location,f"{i}",life_time = 30)