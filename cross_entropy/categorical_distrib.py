#!/usr/bin/env python

#*****************************************************
# Author: Matt Litton
# Purpose: Implements a Categorical Distribution for Cross-Entropy
# Last Modified: 10/11/2022
#*****************************************************

"""
Notes:
    --You can explicity provide probabilities for each category in the constructor, 
            if none are provided all are set to equal probabilities
"""

from numpy.random import default_rng
import numpy as np

class CategoricalDistrib(object):
    def __init__(self, categories, probabilities=None):
        self.rng = default_rng()
        self.categories = categories
        if probabilities is None:
            self.probabilities = (np.ones(len(categories)) * 1/len(categories)).tolist()
        else: self.probabilities = probabilities
    
    def draw_samples(self, N):
        return self.rng.choice(self.categories, N, self.probabilities)
    
    def update_params(self, elites):
        num_elites = len(elites)
        for i in range(len(self.categories)):
            self.probabilities[i] = elites.tolist().count(self.categories[i])/num_elites
    
    def print_params(self):
        print(f"Categories:{self.categories},Probabilities:{self.probabilities}")