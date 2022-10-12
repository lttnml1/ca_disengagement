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

from scenarios.ScenarioClass import Scenario
from util import helper_functions as hf

from agents.basic_agent import BasicAgent
import carla
import pandas as pd
import csv
import os


class Scenario_CutIn(Scenario):
    def __init__(self, host, port, parameters, args):
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
                        if (self.adv.get_transform().location - self.ego.get_transform().location).x > self.distance_when_lane_change:
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
                    self.score += 0.1 * (1/stopped_frames['distance'].min())
                else:
                    self.score -= .1 * (df_moving['distance'].min())

        return self.score
    
    def write_path(self):
        path_file_name = self.file_name.split('.csv')[0] + "_path.csv"
        with open(os.path.join(self.data_path,path_file_name),'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ego_target_speed','adversary_speed_differential','distance_when_lane_change','lane_offset'])
            writer.writerow([self.ego_target_speed, self.adversary_speed_differential,self.distance_when_lane_change,self.lane_offset])



        

