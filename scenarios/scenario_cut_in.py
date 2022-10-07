#!/usr/bin/env python
"""
    Parameters:
        [0] adversary_target_speed       (normal)

        [1] ego_target_speed             (normal)

        [2] distance_when_lane_change    (normal)
            distance in front of the ego when adversary executes lane change

"""


from configparser import BasicInterpolation
from scenarios.ScenarioClass import Scenario
from util import helper_functions as hf
import random

from agents.basic_agent import BasicAgent
import carla


class Scenario_CutIn(Scenario):
    def __init__(self, host, port,parameters):
        super().__init__(host, port)
        self.world_setup()

        self.ego = None
        self.adv = None

        self.adversary_target_speed = parameters[0]
        self.ego_target_speed = parameters[1]
        self.distance_when_lane_change = parameters[2]

        self.features = []
        self.score = None
    
    def execute_scenario(self):
        try: 
            
            ego_blueprint = self.blueprints.filter("vehicle.dodge.charger_police")[0]
            ego_spawn_point = self.spawn_points[4]
            self.ego = self.world.try_spawn_actor(ego_blueprint,ego_spawn_point)

            adversary_blueprint = self.blueprints.filter("vehicle.tesla.model3")[0]
            adversary_blueprint.set_attribute('color','200,0,0')
            self.adv = self.world.try_spawn_actor(adversary_blueprint,self.spawn_points[3])

            self.world.tick()

            ego_agent = BasicAgent(self.ego,target_speed=self.ego_target_speed)
            adv_agent = BasicAgent(self.adv,target_speed=self.adversary_target_speed)

            changed_lanes = False
            counter = 0
            while True:
                try:
                    self.world.tick()
                    self.ego.apply_control(ego_agent.run_step())
                    self.adv.apply_control(adv_agent.run_step())
                    
                    if(not changed_lanes):
                        if (self.adv.get_transform().location - self.ego.get_transform().location).x > self.distance_when_lane_change:
                            adv_loc = self.adv.get_transform().location
                            adv_waypoint = self.map.get_waypoint(adv_loc)
                            right_lane_loc = adv_waypoint.get_right_lane().next(10)
                            adv_agent.set_destination(right_lane_loc[0].transform.location)
                            changed_lanes = True
                            new_wp = self.map.get_waypoint(carla.Location(x=84.220612, y=207.268448, z=0.952669))
                            adv_agent.set_destination(new_wp.transform.location)
                    elif(changed_lanes):
                        if adv_agent.done():
                            break
                    if(counter > 1000):
                        print("Timeout")
                        break
                    counter += 1
                        
                    
                except KeyboardInterrupt:
                    print("Interrupted by user!")
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
    
    def score_scenario(self):
        return self.score



        

