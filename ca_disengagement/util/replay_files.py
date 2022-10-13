#!/usr/bin/env python
import os
import random
import time

from ca_disengagement.scenarios.scenario_cut_in import Scenario_CutIn

def replay():
    replay_path = os.path.join(os.getcwd(),"ca_disengagement\\data")
    for dirName, subdirList, fileList in os.walk(replay_path):
        good_files = [f for f in fileList if '0_path' in f]
        bad_files = [f for f in fileList if '1_path' in f]
    
    num_to_sample = 3
    good_files_subset = random.sample(good_files,num_to_sample)
    bad_files_subset = random.sample(bad_files,num_to_sample)
    
    
    counter = 0
    for file in good_files_subset:
        print(counter, file)
        ret = Scenario_CutIn.replay(os.path.join(replay_path,file))
        if(ret < 0):
            print("Good files replay loop interrupted by user")
            return
        counter +=1

    time.sleep(10)
    counter = 0            
    for file in bad_files_subset:
        print(counter, file)
        ret = Scenario_CutIn.replay(os.path.join(replay_path,file))
        if(ret < 0):
            print("Bad files replay loop interrupted by user")
            return
        counter +=1
    