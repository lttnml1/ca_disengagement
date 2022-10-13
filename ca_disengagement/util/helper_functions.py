#!/usr/bin/env python

def draw_spawn_point_locations(world, spawn_points):
    for i in range(len(spawn_points)):
        world.debug.draw_string(spawn_points[i].location,f"{i}",life_time = 30)

def draw_location(world, location):
    world.debug.draw_point(location,life_time = 30)