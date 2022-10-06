#!/usr/bin/env python
"""
    Parameters:
        [0] adversary_target_speed       (normal)

        [1] ego_target_speed             (normal)

        [2] distance_when_lane_change    (normal)
            distance in front of the ego when adversary executes lane change

"""


from scenarios.ScenarioClass import Scenario
from util import helper_functions as hf
import random


class Scenario_CutIn(Scenario):
    def __init__(self, host, port,parameters):
        super().__init__(host, port)
        self.world_setup()

        self.ego = None
        self.adversary = None

        self.adversary_target_speed = parameters[0]
        self.ego_target_speed = parameters[1]
        self.distance_when_lane_change = parameters[2]
    
    def execute_scenario(self):
        try: 
            ego_blueprint = self.blueprints.filter("vehicle.dodge.charger_police")[0]
            self.ego = self.world.try_spawn_actor(ego_blueprint,self.spawn_points[4])
            adversary_blueprint = self.blueprints.filter("vehicle.tesla.model3")[0]
            adversary_blueprint.set_attribute('color','200,0,0')
            self.adversary = self.world.try_spawn_actor(adversary_blueprint,self.spawn_points[3])

            self.world.tick()


            while True:
                try:
                    self.world.tick()
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
                if self.adversary is not None:
                    self.adversary.destroy()



        

