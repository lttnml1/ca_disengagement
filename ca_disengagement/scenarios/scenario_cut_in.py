#!/usr/bin/env python
"""
    Parameters:
        [0] adversary_speed_differential (normal)
            how much faster the adversary is going than the ego

        [1] distance_when_lane_change    (normal)
            distance in front of the ego when adversary executes lane change
        
        [2] lane_offset                  (categorical)
            adversary's offset within the lane
            -1:     far left in lane
            -0.5:   left of center
            0:      center of the lane
            0.5:    right of center
            1:      far right in lane

"""

from abc import abstractmethod
from ca_disengagement.scenarios.scenario_base import Scenario
from util import helper_functions as hf

from agents.basic_agent import BasicAgent
import carla
import pandas as pd
import csv
import os
import random
import time

from ca_disengagement.agents import misc


class Scenario_CutIn(Scenario):
    def __init__(self, host, port, parameters, args=None):
        super().__init__(host, port, args)
        self.Town = 'Town03'
        self.init = self.world_setup()

        self.ego = None
        self.adv = None

        self.ego_target_speed = 8

        self.adversary_speed_differential = parameters[0]
        self.distance_when_lane_change = parameters[1]
        self.lane_offset = parameters[2]

        self.features = []
        self.completed = False
        
    
    def check_for_bad_path(self):
        self.score = 0
        if (self.adversary_speed_differential <= 0):
            self.score += 10 * abs(self.adversary_speed_differential)
            self.badPath = True
        if (self.distance_when_lane_change < 0):
            self.score += 20 * abs(self.distance_when_lane_change)
            self.badPath = True
        if abs(self.lane_offset) > 1:
            self.score += 10 * abs(self.lane_offset)
            self.badPath = True            

    def execute_scenario(self):
        exit_code = 0
        try: 
            if(not self.no_render):
                spectator = self.world.get_spectator()
                spectator.set_transform(carla.Transform(carla.Location(x=45, y=180, z=50),carla.Rotation(roll=0,pitch=-70,yaw=90)))
            
            ego_blueprint = self.blueprints.filter("vehicle.dodge.charger_police")[0]
            ego_blueprint.set_attribute('role_name','ego')
            ego_spawn_point = self.spawn_points[4]
            self.ego = self.world.try_spawn_actor(ego_blueprint,ego_spawn_point)

            adversary_blueprint = self.blueprints.filter("vehicle.tesla.model3")[0]
            adversary_blueprint.set_attribute('role_name', 'adversary')
            adversary_blueprint.set_attribute('color','200,0,0')
            self.adv = self.world.try_spawn_actor(adversary_blueprint,self.spawn_points[3])

            self.world.tick()

            ego_agent = BasicAgent(self.ego,target_speed=self.ego_target_speed)
            adv_agent = BasicAgent(self.adv,target_speed=self.ego_target_speed + self.adversary_speed_differential,opt_dict={'offset':self.lane_offset})

            started_lane_change = False
            finished_lane_change = False
            
            timeout_counter = 0
            crash_counter = 0
            while True:
                try:
                    self.world.tick()
                    new_wp = None

                    if(self.get_features()[1]):
                        crash_counter += 1
                    else: crash_counter = 0

                    if(crash_counter > 50):
                        print("Crash")
                        break
                    elif(timeout_counter > 1000):
                        print("Timeout")
                        break
                    elif(not started_lane_change):
                        #get front of ego_vic
                        ego_transform = self.ego.get_transform()
                        ego_forward_vector = ego_transform.get_forward_vector()
                        ego_extent = self.ego.bounding_box.extent.x
                        ego_front_transform = ego_transform
                        ego_front_transform.location += carla.Location(
                            x=ego_extent * ego_forward_vector.x,
                            y=ego_extent * ego_forward_vector.y,
                        )
                        
                        if (self.adv.get_transform().location - ego_front_transform.location).x > self.distance_when_lane_change:
                            adv_loc = self.adv.get_transform().location
                            adv_waypoint = self.map.get_waypoint(adv_loc)
                            right_lane_loc = adv_waypoint.get_right_lane()
                            adv_agent.set_destination(right_lane_loc.transform.location)
                            started_lane_change = True
                    elif(not finished_lane_change):
                        if(adv_agent.done()):
                            finished_lane_change = True
                            new_wp = self.map.get_waypoint(carla.Location(x=84.220612, y=207.268448, z=0.952669))
                            adv_agent.set_destination(new_wp.transform.location)
                            ego_agent.set_destination(new_wp.transform.location)
                    elif(finished_lane_change):
                        if(adv_agent.is_done() or ego_agent.is_done() or adv_agent.done() or ego_agent.done()):
                            self.completed = True
                            break
                    
                    self.ego.apply_control(ego_agent.run_step())
                    self.adv.apply_control(adv_agent.run_step())
                    
                    timeout_counter += 1
                        
                except KeyboardInterrupt:
                    print("Scenario Loop Interrupted By User!")
                    exit_code = -1
                    break
        
        finally:    
            if self.world is not None:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
                if self.ego is not None:
                    self.ego.destroy()
                if self.adv is not None:
                    self.adv.destroy()
            return exit_code
    
    def execute_scenario_2(self):
        exit_code = 0
        try: 
            ego_blueprint = self.blueprints.filter("vehicle.dodge.charger_police")[0]
            ego_blueprint.set_attribute('role_name','ego')
            spawn_points = self.find_good_spawn_points()
            if(spawn_points):
                sp = random.choice(spawn_points)
                ego_spawn_point = self.spawn_points[sp[0]]
                adv_spawn_point = self.spawn_points[sp[1]]
                lange_change = sp[2]
            self.ego = self.world.try_spawn_actor(ego_blueprint,ego_spawn_point)
            
            adversary_blueprint = self.blueprints.filter("vehicle.tesla.model3")[0]
            adversary_blueprint.set_attribute('role_name','adversary')
            adversary_blueprint.set_attribute('color','200,0,0')
            self.adv = self.world.try_spawn_actor(adversary_blueprint,adv_spawn_point)

            print(f"Adversary will be executing a {lange_change} maneuver")
            
            for i in range(10):
                self.world.tick()
            

            if(not self.no_render):
                spectator = self.world.get_spectator()
                spectator.set_transform(carla.Transform(ego_spawn_point.location + carla.Location(z=50),carla.Rotation(roll=0,pitch=-50,yaw=0)))

            ego_agent = BasicAgent(self.ego,target_speed=self.ego_target_speed)
            adv_agent = BasicAgent(self.adv,target_speed=self.ego_target_speed + self.adversary_speed_differential,opt_dict={'offset':self.lane_offset})

            started_lane_change = False
            finished_lane_change = False
            
            timeout_counter = 0
            crash_counter = 0
            
            
            
            while True:
                try:
                    self.world.tick()
                    new_wp = None

                    if(self.get_features()[1]):
                        crash_counter += 1
                    else: crash_counter = 0

                    if(crash_counter > 50):
                        print("Crash")
                        break
                    elif(timeout_counter > 1000):
                        print("Timeout")
                        break
                    elif(not started_lane_change):
                        #get front of ego_vic
                        ego_transform = self.ego.get_transform()
                        ego_forward_vector = ego_transform.get_forward_vector()
                        ego_extent = self.ego.bounding_box.extent.x
                        ego_front_transform = ego_transform
                        ego_front_transform.location += carla.Location(
                            x=ego_extent * ego_forward_vector.x,
                            y=ego_extent * ego_forward_vector.y,
                        )
                        
                        if (self.adv.get_transform().location - ego_front_transform.location).x > self.distance_when_lane_change:
                            adv_loc = self.adv.get_transform().location
                            adv_waypoint = self.map.get_waypoint(adv_loc)
                            if lange_change == 'CHANGE_RIGHT':
                                new_dest = adv_waypoint.get_right_lane().transform.location
                                adv_agent.set_destination(new_dest)
                            elif lange_change == 'CHANGE_LEFT':
                                new_dest = adv_waypoint.get_left_lane().transform.location
                                adv_agent.set_destination(new_dest)
                            else:
                                print("Not a valid lane change maneuver")
                                break
                            started_lane_change = True
                            print("started lane change")
                            hf.draw_location(self.world,new_dest,life_time=5)
                    elif(not finished_lane_change):
                        if(adv_agent.done()):
                            finished_lane_change = True
                            print("finished lane change")
                            new_wp = self.map.get_waypoint(self.ego.get_transform().location,project_to_road=True,lane_type=carla.LaneType.Driving).next_until_lane_end(5)[5]
                            hf.draw_location(self.world,new_wp.transform.location, life_time = 30)
                            adv_agent.set_destination(new_wp.transform.location)
                            ego_agent.set_destination(new_wp.transform.location)
                    elif(finished_lane_change):
                        if(adv_agent.is_done() or ego_agent.is_done() or adv_agent.done() or ego_agent.done()):
                            self.completed = True
                            print("completed scenario")
                            break
                    
                    self.ego.apply_control(ego_agent.run_step())
                    self.adv.apply_control(adv_agent.run_step())
                    
                    timeout_counter += 1
                        
                except KeyboardInterrupt:
                    print("Scenario Loop Interrupted By User!")
                    exit_code = -1
                    break
        finally:    
            if self.world is not None:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
                if self.ego is not None:
                    self.ego.destroy()
                if self.adv is not None:
                    self.adv.destroy()
            return exit_code

    def find_good_spawn_points(self):
        # 1) Speed limit is above 90 km/h
        # 2) There is sufficient distance to the end of the highway for a passing maneuver
        # 3) There is either a left or right equivalent to the ego_spawn_point

        over_90 = []
        with_space = []
        good_ego_spawn_points = []
        for i in range(len(self.spawn_points)):
            ego_spawn_point = self.spawn_points[i]
            wp = self.map.get_waypoint(ego_spawn_point.location,project_to_road=True,lane_type=carla.LaneType.Driving)
            if(self.speed_limit_over_90(wp)):
                over_90.append(i)
                if(self.distance_to_end(wp)):
                    with_space.append(i)
                    possible_lane_changes = self.lane_change(wp)
                    if(possible_lane_changes):
                        for p in possible_lane_changes:
                            good_ego_spawn_points.append((i,p[0],p[1]))
        return good_ego_spawn_points
    
    def speed_limit_over_90(self, wp):
        speed_limit_signs = wp.get_landmarks_of_type(100,'274')
        if speed_limit_signs:
            for sign in speed_limit_signs:
                if sign.value >= 90.0: 
                    return True
        else: return False

    def distance_to_end(self, wp):
        lane_waypoints = wp.next_until_lane_end(5)
        if(len(lane_waypoints) >= 20): return True
        else: return False
    
    def lane_change(self, wp):
        possible = []

        left_wp = wp.get_left_lane()
        left_lane_id = left_wp.lane_id
        right_wp = wp.get_right_lane()
        right_lane_id = right_wp.lane_id
        if right_wp is not None:
            #see if there's a spawn point close by
            for i in range(len(self.spawn_points)):
                sp_to_wp = self.map.get_waypoint(self.spawn_points[i].location,project_to_road=True,lane_type=carla.LaneType.Driving)
                if(right_lane_id == sp_to_wp.lane_id and misc.compute_distance(right_wp.transform.location,sp_to_wp.transform.location) < 25):
                    possible.append((i, 'CHANGE_LEFT'))
        if left_wp is not None:
            #see if there's a spawn point close by
            for i in range(len(self.spawn_points)):
                sp_to_wp = self.map.get_waypoint(self.spawn_points[i].location,project_to_road=True,lane_type=carla.LaneType.Driving)
                if(left_lane_id == sp_to_wp.lane_id and misc.compute_distance(left_wp.transform.location,sp_to_wp.transform.location) < 25):
                    possible.append((i, 'CHANGE_RIGHT'))
        
        return possible

    def score_scenario_ONES(self):
        headers = ['frame', 'intersect','distance','angle', 
                   'ego_vel_x','ego_vel_y','ego_vel_z','ego_accel_x','ego_accel_y','ego_accel_z','ego_ang_vel_x','ego_ang_vel_y','ego_ang_vel_z',
                   'adv_vel_x','adv_vel_y','adv_vel_z','adv_accel_x','adv_accel_y','adv_accel_z','adv_ang_vel_x','adv_ang_vel_y','adv_ang_vel_z']
        self.dataframe = pd.DataFrame(data = self.feature_vector, columns=headers)

        self.score = 0
        df_moving = self.dataframe[25:]

        #was there a crash?
        crash = df_moving[df_moving['intersect'] == True].shape[0]
        if(crash > 0):
            self.score -= crash/10
            return self.score
        else:
            #did the scenario complete?
            if(not self.completed):
                self.score += 10 * abs(df_moving['angle'].min())
                return self.score
            else:
                #did the ego have to stop?
                stopped_frames = df_moving[df_moving['ego_vel_x'] < 0.1]
                num_frames_stopped = stopped_frames.shape[0]
                if(num_frames_stopped > 0):
                    self.score -= 0.1 * (1/stopped_frames['distance'].min())

        return self.score

    def score_scenario_ZEROS(self):
        headers = ['frame', 'intersect','distance','angle', 
                'ego_vel_x','ego_vel_y','ego_vel_z','ego_accel_x','ego_accel_y','ego_accel_z','ego_ang_vel_x','ego_ang_vel_y','ego_ang_vel_z',
                'adv_vel_x','adv_vel_y','adv_vel_z','adv_accel_x','adv_accel_y','adv_accel_z','adv_ang_vel_x','adv_ang_vel_y','adv_ang_vel_z']
        self.dataframe = pd.DataFrame(data = self.feature_vector, columns=headers)

        self.score = 0
        df_moving = self.dataframe[25:]

        #was there a crash?
        crash = df_moving[df_moving['intersect'] == True].shape[0]
        if(crash > 0):
            self.score += crash/10
            return self.score
        else:
            #did the scenario complete?
            if(not self.completed):
                self.score += 10 * abs(df_moving['angle'].min())
                return self.score
            else:
                #did the ego have to stop?
                stopped_frames = df_moving[df_moving['ego_vel_x'] < 0.1]
                num_frames_stopped = stopped_frames.shape[0]
                if(num_frames_stopped > 0):
                    #self.score += 0.1 * (1/stopped_frames['distance'].min())
                    self.score -= 1/num_frames_stopped #it's ok that the ego stopped, but less stopping the better (i.e. that's why we subtract and do 1/num_frames_stopped)
                else:
                    self.score -= .1 * (df_moving['distance'].min())

        return self.score
    
    def write_path(self):
        path_file_name = self.file_name.split('.csv')[0] + "_path.csv"
        with open(os.path.join(self.data_path,path_file_name),'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ego_target_speed','adversary_speed_differential','distance_when_lane_change','lane_offset'])
            writer.writerow([self.ego_target_speed, self.adversary_speed_differential,self.distance_when_lane_change,self.lane_offset])
    
    @abstractmethod
    def replay(file):
        print(f"Replaying {file}")
        #read in the file - extract parameters
        df = pd.read_csv(file)
        params = [df['adversary_speed_differential'][0],df['distance_when_lane_change'][0],df['lane_offset'][0]]
        print(params)
        #create a new object with those parameters
        replay_scenario = Scenario_CutIn('127.0.0.1',2004,params)
        
        #execute the scenario
        ret = replay_scenario.execute_scenario_2()
        if(ret < 0):
            print("Replay Interrupted By User!")
            return -1
        return 0



        

