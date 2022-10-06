#!/usr/bin/env python

from scenarios.scenario_cut_in import Scenario_CutIn


"""
    "insufficient yielding for cut-in"
    This file should
        1) specify the scenario to use
        2) execute cross-entropy (with the specified cost function) to generate 100 0's and 100 1's
        3) train/test a RDF classifier with that data
"""

def main():
    cut_in = Scenario_CutIn('127.0.0.1',2008,[10,10,10])
    cut_in.execute_scenario()

if __name__ == '__main__':
    main()