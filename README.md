# Purpose
This code trains a RDF classifier on good/bad behavior of an autonomous vehicles using scenarios derived from the California DMV disengagement reports.

# Structure
[agents/](https://github.com/lttnml1/ca_disengagement/tree/main/agents): Contains the code which controls the behavior of autonomous vehicle (ego) and adversaries.  
[cross_entropy/](https://github.com/lttnml1/ca_disengagement/tree/main/cross_entropy): Contains code to define various distributions as well as the cross entropy class which defines the primary cross-entropy loop.  
[scenarios/](https://github.com/lttnml1/ca_disengagement/tree/main/scenarios): Defines scenarios that inherit from the ScenarioClass base class.  
[main.py](https://github.com/lttnml1/ca_disengagement/blob/main/main.py): The main program that generates the 0's/1's  
[process_labeled.ipynb](https://github.com/lttnml1/ca_disengagement/blob/main/process_labeled.ipynb): Jupyter notebook that processes the data generated and trains/evaluates the RDF classifier  

