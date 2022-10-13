#!/usr/bin/env python

import argparse
from ca_disengagement.cross_entropy.categorical_distrib import CategoricalDistrib
from ca_disengagement.cross_entropy.normal_distrib import NormalDistrib
import time

"""
    Need to add 'agents' and 'cross_entropy' to the search path
    To see what's in the current path run '>> pprint.pprint(sys.path)'
"""

from ca_disengagement.cross_entropy.cross_entropy import CrossEntropy

"""
    This file should
        1) execute cross-entropy to generate 100 0's and 100 1's
"""

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--no_render',
        action = 'store_true',
        help='Render graphics (default: False)')
    args = argparser.parse_args()

    program_start_time = time.time() 

    #generate 0's and 1's
    for i in range(10):
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


        print(f"*****Beginning Round {i} in search for ONES*****")
        round_start_time = time.time()    
        ce = CrossEntropy(10,0.1,0,distributions)
        ret = ce.execute_ce_search(args, 'ONES')
        if ret < 0:
            print("Main Loop Interrupted!")
            break
        print(f"\n*****Round: {i} took {time.time()-round_start_time}*****")

        
        
    #*******************************************************************************    
        
        
        distributions = []
    
        #Distribution 1 (Normal)
        #   Difference between adversary speed and ego speed
        adversary_speed_differential = NormalDistrib(5,1)
        distributions.append(adversary_speed_differential)

        #Distribution 2 (Normal)
        #   Distance that adv is ahead of ego before attempting lane change
        distance_when_lane_change = NormalDistrib(4,1)
        distributions.append(distance_when_lane_change)

        #Distribution 3 (Categorical)
        #   Adversary's offset within the lane
        lane_offset = CategoricalDistrib([-1,-0.5,0,0.5,1])
        distributions.append(lane_offset)

        print(f"*****Beginning Round {i} in search for ZEROS*****")
        round_start_time = time.time()    
        ce = CrossEntropy(10,0.1,0,distributions)
        ret = ce.execute_ce_search(args, 'ZEROS')
        if ret < 0:
            print("Main Loop Interrupted!")
            break
        print(f"\n*****Round: {i} took {time.time()-round_start_time}*****")
    
    print("*****************************************************************")
    print("*****************************************************************")
    print(f"\nENTIRE PROGRAM TOOK {time.time()-program_start_time}")
    print("*****************************************************************")
    print("*****************************************************************")

if __name__ == '__main__':
    #main()
    from ca_disengagement.util import replay_files
    replay_files.replay()