#!/usr/bin/env python

#*****************************************************
# Author: Matt Litton
# Purpose: Cross-Entropy Implementation
# Last Modified: 10/11/2022
#*****************************************************

"""
Notes:
    --The Scenario type is specified in the main CE body ('execute_ce_search()') i.e.,
        ********************************************************
        scenario = Scenario_CutIn('127.0.0.1',2004,y[i,:], args)
        scenario.check_for_bad_path()
        if(not scenario.badPath):
            scenario.execute_scenario()
            scenario.score_scenario()
        ********************************************************
"""

import numpy as np
import time
from scenarios.scenario_cut_in import Scenario_CutIn

class CrossEntropy(object):
    def __init__(self, N, rho, gamma, distributions):
        self.N = N
        self.rho = rho
        self.gamma = gamma
        self.distributions = distributions
        self.n = len(distributions)
    
    def draw_random_samples(self, num_samples=None):
        if num_samples is None:
            num_samples = self.N
        y=np.empty((num_samples,self.n))
        for i in range(self.n):
            y[:,i] = self.distributions[i].draw_samples(num_samples)
        return y
    
    def execute_ce_search(self, args):
        gamma = 100
        round = 0
        while(gamma > self.gamma):
            print(f"*****Beginning Round {round}*****")
            round_start_time = time.time()
            y = self.draw_random_samples()
            scores = np.empty(y.shape)
            for i in range(np.shape(y)[0]):
                scenario = Scenario_CutIn('127.0.0.1',2004,y[i,:], args)
                scenario.check_for_bad_path()
                if(not scenario.badPath):
                    ret = scenario.execute_scenario()
                    if(ret < 0):
                        print("Cross-Entropy Loop Interrupted By User!")
                        return -1
                    scenario.score_scenario()
                scores[i,0] = scenario.score
                print(f"Completed\t{i+1}/{np.shape(y)[0]}")
            
            gamma, elites = self.calculate_elite(y, scores)
            self.update_parameters(elites)
            
            self.print_distribution_parameters()
            print(f"Gamma:{gamma}")
            print(f"\n*****Round: {round} took {time.time()-round_start_time}*****")
            round += 1
    
    def calculate_elite(self, y, scores, debug=False):
        distribution_elites = []
        if(debug): 
            print("Original scores/values are:")
            for i in range(len(y)):
                print(f"{i}:\t{y[i,:]}\t{scores[i,0]}")
        sorted_scores = np.sort(scores[:,0])
        sorted_score_indices = np.argsort(scores[:,0])
        if(debug): print(f"Sorted scores are: {sorted_scores}, indices are: {sorted_score_indices}")
        
        gamma_index = round(self.rho * self.N)

        if(debug): print(f"The gamma element should be at {gamma_index+1}, which means it's {sorted_scores[gamma_index+1]}")
        gamma_element = sorted_scores[gamma_index+1]
        
        elite_set = sorted_score_indices[:gamma_index+2]

        if(debug): print(f"Elite set has indicies: {elite_set}")

        for i in range(self.n):
            if(debug):
                print(f"**For distribution: {i} **")
                print(f"This means the elite set is made up of: {y[elite_set,i]}")
            distribution_elite = y[elite_set,i]
            distribution_elites.append(distribution_elite)
        return gamma_element, distribution_elites
    
    def update_parameters(self, elites):
        for i in range(self.n):
            self.distributions[i].update_params(elites[i])
    
    def print_distribution_parameters(self):
        for i in range(self.n):
            self.distributions[i].print_params()
    
    

        

