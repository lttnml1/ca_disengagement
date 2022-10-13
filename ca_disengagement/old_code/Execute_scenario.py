#!/usr/bin/env python

# Author:           Matthew Litton
# Last Modified:    8/5/2022
# Purpose:          Main script for running/retrieving data from simulation for Cross-Entropy - supposed to be as general as possible

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import math
import os
import numpy.random as random
import re
import sys
import weakref
import csv
import rtamt

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from ca_disengagement.agents.basic_agent import BasicAgent
from ca_disengagement.agents.simple_agent import SimpleAgent
from ca_disengagement.agents.behavior_agent import BehaviorAgent

from enum import Enum
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from rtamt.spec.stl.discrete_time.specification import Semantics

# ==============================================================================
# -- Helper Functions ----------------------------------------------------------
# ==============================================================================

def get_2D_distance(loc1, loc2):
    return math.sqrt((loc1.x - loc2.x)**2+(loc1.y-loc2.y)**2)

def get_vehicle_speed(actor):
    """Returns vehicle speed in km/hr"""
    vel = actor.get_velocity()
    speed = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)#m/s * 1km/1000m * 3600s/1hr = km/hr (i.e., m/s * 3.6 = km/hr, 3.6 = (km*s)/(m*hr))
    return speed

def read_csv(filename):
    f = open(filename, 'r')
    reader = csv.reader(f)
    headers = next(reader, None)

    column = {}
    for h in headers:
        column[h] = []

    for row in reader:
        for h, v in zip(headers, row):
            column[h].append(float(v))

    return column



# ==============================================================================
# -- Scenario ------------------------------------------------------------------
# ==============================================================================

class Scenario(object):

    def __init__(self, args):
        self._args = args
        self.file = args.file
        self.score = 0
        self.point_array = []
        self.speed_array = []
        self.accel_array = []
        self.destination_array = []
        self.frame = 0
        self.accident = False
        self.fault = "nobody"
        self.ego_start, self.ego_end = Scenario.get_random_start_end_for_ego()
        self.read_file()

    @staticmethod
    def get_random_start_end_for_ego():
        start_points_dict = {'start_points':[15,228,61,98]}
        end_points_dict = {}
        end_points_dict.update({15:[229,58,9,90,91]})
        end_points_dict.update({228:[58,9,90,91,31]})
        end_points_dict.update({61:[229,31,90,91,9]})
        end_points_dict.update({98:[58,229,31,90,91]})
            
        start_point = random.choice(start_points_dict['start_points'])
        end_point = random.choice(end_points_dict[start_point])
        return start_point,end_point

    def read_file(self):
        with open(self.file, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                self.point_array.append(row[1])
                self.speed_array.append(float(row[2]) * 3.6)
                self.accel_array.append(float(row[3]))
    
    def score_path(self):
        isBadPath = False
        #if(self.point_array[-1] != '311'): 
        #    isBadPath = True
        #    self.score = 999
        #else:
        #self.score -= 1/len(self.point_array)
        if(self.accel_array[0]<0):
            isBadPath = True
            self.score += 100*abs(self.accel_array[0])
        else:
            for i in range(0,len(self.point_array)):
                speed = self.speed_array[i]
                #no NaN speed
                if math.isnan(speed):
                    isBadPath = True
                    #self.score = sys.float_info.max/2000
                    self.score += 999
                    break
                #speed is non-negative    
                elif speed<0:
                    isBadPath = True
                    self.score += 100*abs(speed)
                    break
        if(self._args.debug_score): print(f"isBadPath:\t{isBadPath}, score:\t{self.score}")
        return isBadPath

# ==============================================================================
# -- Obstacle Sensor -----------------------------------------------------------
# ==============================================================================
class ObstacleSensor(object):

    def __init__(self, parent_actor):       
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.obstacle')
        blueprint.set_attribute('distance','5')
        blueprint.set_attribute('hit_radius','0.5')
        blueprint.set_attribute('only_dynamics','True')
        #blueprint.set_attribute('debug_linetrace','True')
        blueprint.set_attribute('sensor_tick','0.05')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: ObstacleSensor._on_obstacle_detected(weak_self, event))
        self.history = {'frame':[],'distance':[]}

    @staticmethod
    def _on_obstacle_detected(weak_self, event):
        self = weak_self()
        if not self:
            return
        #print(f"Obstacle detected by {event.actor.parent} at frame {event.frame}:\t{event.other_actor} at {event.distance}")
        self.history['frame'].append(event.frame)
        self.history['distance'].append(event.distance)

# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):

    def __init__(self, carla_world, args, grid):
        self.world = carla_world
        self._args = args
        self._grid = grid

        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        
        self.ego = None
        self.obstacle_sensor_ego = None
        self.adversary = None
        self.feature_vector = []
    
    def destroy(self):
        actors = [
            self.adversary,
            self.ego]
        for actor in actors:
            if actor is not None:
                if actor == self.ego:
                    self.obstacle_sensor_ego.sensor.destroy()
                actor.destroy()

    def draw_location_on_grid(self, location, draw_time = 10):
        self.world.debug.draw_point(location,size=0.2,color=carla.Color(0,255,0),life_time=draw_time)
            
    def draw_grid(self, draw_time = 10):
        #draw vertical lines
        y=self._grid.left
        for i in range(0,21):
            self.world.debug.draw_line(carla.Location(x=self._grid.top,y=y,z=1), carla.Location(x=self._grid.bottom,y=y,z=1), thickness=0.1, color=carla.Color(255,0,0), life_time=draw_time)
            y+=self._grid.box_width
        #draw horizontal lines
        x=self._grid.bottom
        for i in range(0,21):
            self.world.debug.draw_line(carla.Location(x=x,y=self._grid.left,z=1), carla.Location(x=x,y=self._grid.right,z=1), thickness=0.1, color=carla.Color(255,0,0), life_time=draw_time)
            x+=self._grid.box_height

    def convert_points_to_locations(self, scenario: Scenario):
        for point in scenario.point_array:
            i, j = self._grid.return_coords_from_point(point) 
            dest = self._grid.return_location_from_grid(i,j)
            scenario.destination_array.append(dest)
    
    def draw_points_and_locations(self, points):
        counter = 0
        for point in points:
            self.world.debug.draw_point(point.location,size=0.2,color=carla.Color(255,255,255),life_time=30)
            #self.world.debug.draw_string(point.location + carla.Location(z=3),str(point.location.x)+',\n'+str(point.location.y),color=carla.Color(255,0,0),life_time=30)
            self.world.debug.draw_string(point.location + carla.Location(z=3),str(counter),color=carla.Color(255,0,0),life_time=30)
            counter += 1
    
    def get_features(self):
        snapshot = self.world.get_snapshot()
        frame = snapshot.frame
        actors = []
        ActorSnapshot_ego = snapshot.find(self.ego.id)
        actors.append(ActorSnapshot_ego)
        ActorSnapshot_adv = snapshot.find(self.adversary.id)
        actors.append(ActorSnapshot_adv)

        frame_feature_vector = []
        frame_feature_vector.append(frame)
        for actor in actors:
            frame_feature_vector.append(actor.get_transform().location.x)
            frame_feature_vector.append(actor.get_transform().location.y)
            vel = actor.get_velocity()
            frame_feature_vector.append(math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2))

        vehicles = []
        vehicles.append(self.ego)
        vehicles.append(self.adversary)
        for vehicle in vehicles:
            control = vehicle.get_control()
            frame_feature_vector.append(control.throttle)
            frame_feature_vector.append(control.steer)
            frame_feature_vector.append(control.brake)
        self.feature_vector.append(frame_feature_vector)

    def write_features(self, scenario, args, frame):
        file_path = "C:\\data\\Features\\"
        header = ["Frame","ego_loc_x","ego_loc_y","ego_velocity","adv_loc_x","adv_loc_y","adv_velocity","ego_throttle","ego_steer","ego_brake","adv_throttle","adv_steer","adv_brake"]
        score = scenario.score
        if(score < 0):
            num = args.file.split('#')[1]
            file_name = f"features_path{num}_label1_score{score:8.6f}_frame{frame}"
            file = file_path + file_name + ".csv"
            with open(file,"w",newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(self.feature_vector)
        elif(score >= 0):
            num = args.file.split('#')[1]
            file_name = f"features_path{num}_label0_score{score:8.6f}_frame{frame}"
            file = file_path + file_name + ".csv"
            with open(file,"w",newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(self.feature_vector)
        else: 
            return

# ==============================================================================
# -- Grid ---------------------------------------------------------------
# ==============================================================================

class Grid(object):
    def __init__(self, top: float, bottom: float, left: float, right: float, draw_time: int = 0):
        #this will be a 20x20 grid
        """
        top/bottom are x values
        left/right are y values
        assume axes look like
        x
        ^
        |
        |
        |
        --------> Y
        """
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.box_width = abs(self.right - self.left)/20
        self.box_height = abs(self.top - self.bottom)/20 

    def return_location_from_grid(self, i: int, j: int, draw_time: int = 0):
        center_point_y = self.left + self.box_width*(j) + self.box_width/2
        center_point_x = self.top - self.box_height*(i) - self.box_height/2
        location = carla.Location(x=center_point_x,y=center_point_y,z=1)
        return location
    
    def return_grid_from_location(self, location):
        i = 0
        j = 0
        for index in range(0,19):
            if(location.x < self.top - self.box_height * (index)): i=index
            if(location.y > self.left + self.box_width* (index)): j=index
        return (i,j)
    
    def return_coords_from_point(self, point):
        i = math.floor(int(point)/20)
        j = int(point) % 20
        return (i,j)
    
    def return_point_from_coords(self, i, j):
        return i * 20 + j

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

def game_loop(args, scenario):

    world = None
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)
        
        sim_world = client.get_world()
        
        settings = sim_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        if(args.no_render): settings.no_rendering_mode = True
        else: settings.no_rendering_mode = False
        sim_world.apply_settings(settings)
        #tm = client.get_trafficmanager(8000)
        #tm.set_synchronous_mode(True)
        #tm.set_random_device_seed(0)

        grid = Grid(-65,-100,-10,30)
        world = World(client.get_world(), args, grid)
        world.convert_points_to_locations(scenario)

        #set the view to the middle of the grid
        spectator = world.world.get_spectator()
        spectator.set_transform(carla.Transform(carla.Location(x=-100,y=14,z=50),carla.Rotation(roll=0, pitch=-70,yaw=0)))

        # Spawn the actors
        blueprints = world.world.get_blueprint_library()

        adversary_blueprint=blueprints.filter("vehicle.diamondback.century")[0]
        adversary_blueprint.set_attribute('role_name', 'adversary')

        ego_blueprint = blueprints.filter("vehicle.dodge.charger_police")[0]
        ego_blueprint.set_attribute('role_name','ego')

        adversary_spawn_point = carla.Transform(scenario.destination_array[0],carla.Rotation(roll=0,pitch=0,yaw=0))

        spawn_points = world.map.get_spawn_points()
        relevant_spawn_points = []
        for sp in spawn_points:
            if (sp.location.x<-30 and sp.location.x>-135 and sp.location.y<65 and sp.location.y>-45):
                relevant_spawn_points.append(sp)
        #world.draw_points_and_locations(spawn_points)
        #ego_spawn_point = spawn_points[15]
        ego_spawn_point = spawn_points[scenario.ego_start]

        world.adversary = world.world.try_spawn_actor(adversary_blueprint,adversary_spawn_point)
        world.ego = world.world.try_spawn_actor(ego_blueprint, ego_spawn_point)
        world.obstacle_sensor_ego = ObstacleSensor(world.ego)
        #world.ego.set_autopilot(True,8000)
        #tm.ignore_lights_percentage(world.ego,100)
        #tm.vehicle_percentage_speed_difference(world.ego,60)

        #This is necessary to ensure vehicle "stabilizes" after "falling"
        for i in range (0,30):
            world.world.tick()

        isScoreable = execute_scenario(world, scenario, spectator)#it's scoreable as long as the adversary didn't get stuck/into an accident

        if isScoreable: score_scenario(world, scenario)        

    finally:
        if(scenario.score > 0 and scenario.fault == "ego"):
            scenario.score = -666666
        print(f"{scenario.score:8.6f}")
        world.write_features(scenario, args, scenario.frame)

        if world is not None:
            #tm.set_synchronous_mode(False)
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            #settings.no_rendering_mode = False
            world.world.apply_settings(settings)

            world.destroy()

# ==============================================================================
# -- execute_scenario() --------------------------------------------------------
# ==============================================================================

def execute_scenario(world, scenario, spectator):
    isScoreable = True
    dest_index = 1

    #set initial destination and target_speed
    destination = scenario.destination_array[dest_index]
    adversary_loca = world.adversary.get_location()
    distance = get_2D_distance(adversary_loca,destination)
    target_speed = 0 + scenario.accel_array[dest_index-1] * (distance/1 * 3.6)  
    adversary_agent = SimpleAgent(world.adversary, destination, target_speed=target_speed)
    ego_agent = BasicAgent(world.ego, target_speed = 9,  opt_dict={'ignore_traffic_lights':'True','base_vehicle_threshold':10.0})
    ego_agent.set_destination(world.map.get_spawn_points()[scenario.ego_end].location)

    big_array = []
    stuck_counter = 0
    while True:
        world.get_features()
        world.world.tick()
        stuck_counter += 1

        adversary_loca = world.adversary.get_location()
        ego_loca = world.ego.get_location()
        adversary_speed = get_vehicle_speed(world.adversary)
        ego_speed = get_vehicle_speed(world.ego)      
        
        small_array = []
        frame = world.world.get_snapshot().timestamp.frame
        small_array.append(frame)
        distance = get_2D_distance(ego_loca,adversary_loca)
        small_array.append(distance)
        small_array.append(ego_speed)
        small_array.append(world.obstacle_sensor_ego.history['frame'].count(int(frame)))
        big_array.append(small_array)


        if(stuck_counter % 200 == 0):
            i,j = world._grid.return_grid_from_location(adversary_loca)
            pt = world._grid.return_point_from_coords(i,j)
            if(pt == int(scenario.point_array[dest_index - 1]) or adversary_speed < 0.1):
                #scenario.score += stuck_counter
                #scenario.score += 1
                #isScoreable = False
                if(world._args.debug_score): 
                    print(f"Exiting because stuck, {pt}\t{scenario.point_array[dest_index - 1]}\nspeed:{adversary_speed}")
                    print(f"Score:{scenario.score}")
                break
            '''
            else:
                print(f"Not stuck, {pt}\t{scenario.point_array[dest_index - 1]}")
                print(f"distance:{distance}\tspeed:{ego_speed}")
            '''
        
        debug_accident = False
        if(world._args.debug_score): 
            debug_accident = True
        if(distance < 3.0):#there's an accident, figure out who's fault it is - we only care if it's the ego's fault
            scenario.accident = True
            ego_detected = False
            if(debug_accident): print(f"Breaking out of loop due to accident at distance: {distance}\tobstacles: {len(world.obstacle_sensor_ego.history['frame'])}")
            #if the ego detected an obstacle in the current frame, it's likely ego's fault
            if len(world.obstacle_sensor_ego.history['frame'])>0:
                last_obstacle_detected = world.obstacle_sensor_ego.history['frame'][-1]
                num_detections = 15 #if ego detected the vehicle for at least this many detections before the accident and didn't stop, it's ego's fault
                if(debug_accident): print(f"Current frame:\t{frame} and speed:\t{ego_speed}, Last obstacle:\t{last_obstacle_detected}")
                if(frame == last_obstacle_detected and len(world.obstacle_sensor_ego.history['frame']) >= num_detections):
                    if((last_obstacle_detected - (num_detections-1)) == world.obstacle_sensor_ego.history['frame'][-num_detections]): 
                        ego_detected = True
                if(ego_detected and ego_speed > 1):
                    #scenario.score += -5
                    scenario.fault = "ego"
                    if(debug_accident):
                        print(f"Accident is the ego's fault: Current frame:\t{frame} and speed:\t{ego_speed}, Last obstacle:\t{last_obstacle_detected}")
                        print(f"Score:{scenario.score}") 
                        for i in range(len(world.obstacle_sensor_ego.history['frame'])):
                            print(f"frame:{world.obstacle_sensor_ego.history['frame'][i]}\tdistance:{world.obstacle_sensor_ego.history['distance'][i]}")
                else:
                    #if ego didn't detect the bike for the last 'num_detections' frames, it's likely the bike's fault
                    scenario.fault = "adversary"
                    if(debug_accident):
                        print(f"Accident is bike's fault - not enough detections prior to accident")
                        for i in range(len(world.obstacle_sensor_ego.history['frame'])):
                            print(f"frame:{world.obstacle_sensor_ego.history['frame'][i]}\tdistance:{world.obstacle_sensor_ego.history['distance'][i]}")
            else:#if ego didn't detect the bike at all, it's likely bike's fault - 'suicide bike'
                scenario.fault = "adversary"
                if(debug_accident):
                    print(f"Accident is bike's fault - ZERO detections prior to accident")
            if(not ego_detected): 
                #scenario.score += -2
                if(debug_accident): print(f"Score:{scenario.score}")
            #isScoreable = False
            break
        
        if adversary_agent.done():
               
            if (dest_index >= len(scenario.destination_array)-1):
                    #print("Adversary's route is complete, breaking out of loop")
                    break
            else:
                dest_index += 1
                stuck_counter = 0
                new_dest = scenario.destination_array[dest_index]
                current_speed = adversary_speed
                distance = get_2D_distance(adversary_loca,new_dest)
                target_speed = current_speed + scenario.accel_array[dest_index-1] * (distance/current_speed * 3.6)  
                #print(f"target speed ({target_speed:4.2f}) = current speed ({current_speed:4.2f} km/hr) + accel ({adversary.accel_array[dest_index-1]:4.2f} km/(hr*s)) * (distance ({distance:4.2f} meters) / speed ({current_speed:4.2f} km\hr) * 3.6 km*sec/(m*hr))")
                adversary_agent.set_destination(new_dest)
                adversary_agent.set_target_speed(target_speed)
        
        if ego_agent.done():
            if(world._args.debug_score): print(f"Ego is done, exiting")
            break

        control = adversary_agent.run_step()
        control.manual_gear_shift = False
        world.adversary.apply_control(control)

        world.ego.apply_control(ego_agent.run_step())
    
    if isScoreable:
        file = "c:\\data\\log_file.csv"
        header = ['time','distance','ego_speed','obstacle_detected']
        with open(file,'w',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)           
            counter = 0
            for i in range(len(world.obstacle_sensor_ego.history['frame'])):
                frame_to_change = world.obstacle_sensor_ego.history['frame'][i]
                new_dist = world.obstacle_sensor_ego.history['distance'][i]
                for small_array in big_array:
                    if(small_array[0]==frame_to_change):
                        small_array[1] = new_dist
            for small_array in big_array:
                if(small_array[-1] == 0 and small_array[1] < 5):
                    small_array[1] = 5.000000
                csvwriter.writerow(small_array)

        
        if len(world.obstacle_sensor_ego.history)>0:
            #overwrite 'ground truth' distance values with detector distance values
            file2 = "c:\\data\\obstacle_detect.csv"
            header2 = ['frame','distance']
            with open(file2,'w',newline='') as csvfile2:
                csvwriter2 = csv.DictWriter(csvfile2, fieldnames = header2)
                csvwriter2.writeheader()
                for i in range(len(world.obstacle_sensor_ego.history['frame'])):
                    csvwriter2.writerow({'frame':world.obstacle_sensor_ego.history['frame'][i],'distance':world.obstacle_sensor_ego.history['distance'][i]})
        
    
    return isScoreable

        

# ==============================================================================
# -- score_scenario() ----------------------------------------------------------
# ==============================================================================

def score_scenario(world, scenario):
    
    
    read_file = 'c:\\data\\log_file.csv'
    write_file = 'c:\\data\\log_file_with_rob.csv'
    dataSet = read_csv(read_file)
    
    spec = rtamt.STLDiscreteTimeSpecification()
    spec.name = 'Test'
    spec.declare_var('distance', 'float')
    spec.declare_var('ego_speed', 'float')
    spec.declare_var('out', 'float')
    spec.set_var_io_type('distance','input')
    spec.set_var_io_type('ego_speed','output')
    
    spec.spec = 'out = always((distance < 5.0)  implies (eventually[0:10](ego_speed < 0.1)))'
    #spec.semantics = Semantics.STANDARD
    spec.semantics = Semantics.OUTPUT_ROBUSTNESS
    
    try:
        spec.parse()
    except rtamt.STLParseException as err:
        print('STL Parse Exception: {}'.format(err))
        sys.exit()

    rob = spec.evaluate(dataSet)
    
    if(len(rob)>0):
        min_rob = rob[0][1]
        if math.isinf(min_rob):
            #min_rob = sys.float_info.max/1000000
            min_rob = 2
            if(world._args.debug_score):
                print(f"Ego never came within 5 meters of adversary")
        else:
            for r in rob:
                if r[1] < min_rob:
                    min_rob = rob[1]
        
        scenario.score += min_rob
        if(world._args.debug_score):
            print(f"Minimum robustness: {str(min_rob)}")
            print(f"Done scoring STL, score is: {scenario.score}")

    
    with open(read_file, mode='r') as inFile, open(write_file, "w",newline='') as outFile:
        csv_reader = csv.reader(inFile)
        csv_writer = csv.writer(outFile)
        headers = next(csv_reader, None)
        headers.append('robustness')
        csv_writer.writerow(headers)
        line_count = 0
        

        changed = False
        for row in csv_reader:
            if(line_count == 0):
                scenario.frame = row[0]
            elif(line_count > 0 and not changed):
                if(rob[line_count][1] != rob[line_count-1][1]): #if the robustness has changed, minimum will always be first
                    scenario.frame = int(row[0])-1
                    changed = True
            new_row = row
            new_row.append(rob[line_count][1])
            csv_writer.writerow(new_row)
            line_count += 1
# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--no_render',
        action = 'store_true',
        help='Render graphics (default: False)')
    argparser.add_argument(
        '--file',
        help='file name to read from',
        default=None,
        type=str)
    argparser.add_argument(
        '--debug_score',
        action = 'store_true',
        help='Debug score (default: False)')

    args = argparser.parse_args()
    
    try:
        #first read the path from the file
        scenario = Scenario(args)
        #ensure path is good - like AbstractScore.java
        if(scenario.score_path()):#if the path is bad, score_path() returns True
            #if the path is bad, output the bad score - save time and don't execute the simulation
            if(args.debug_score): print(f"Bad path, not executing simulation")
            print(f"{scenario.score:8.6f}")
        else: 
            #if the path is good, execute it in simulation
            game_loop(args, scenario)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()