#!/usr/bin/env python

import argparse
import os
import sys
import pathlib
from cross_entropy.categorical_distrib import CategoricalDistrib
from cross_entropy.normal_distrib import NormalDistrib

"""
    Need to add 'agents' and 'cross_entropy' to the search path
    To see what's in the current path run '>> pprint.pprint(sys.path)'
"""

#sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(),"agents"))
from scenarios.scenario_cut_in import Scenario_CutIn

#sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(),"cross_entropy"))
from cross_entropy.CrossEntropy import CrossEntropy

"""
    This file should
        1) execute cross-entropy to generate 100 0's and 100 1's
        2) train/test a RDF classifier with that data
"""

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--no_render',
        action = 'store_true',
        help='Render graphics (default: False)')
    args = argparser.parse_args()

    distributions = []
    
    #Distribution 1 (Normal)
    #   Difference between adversary speed and ego speed
    adversary_speed_differential = NormalDistrib(3,1)
    distributions.append(adversary_speed_differential)

    #Distribution 2 (Normal)
    #   Distance that adv is ahead of ego before attempting lane change
    distance_when_lane_change = NormalDistrib(2,1)
    distributions.append(distance_when_lane_change)

    #Distribution 3 (Categorical)
    #   Adversary's offset within the lane
    lane_offset = CategoricalDistrib([-1,-0.5,0,0.5,1])
    distributions.append(lane_offset)

    #generate 100 1's
    for i in range(100):    
        ce = CrossEntropy(100,0.1,0,distributions)
        ret = ce.execute_ce_search(args)
        if ret < 0:
            print("Main Loop Interrupted By User!")
            break
    
if __name__ == '__main__':
    

    main()