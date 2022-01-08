# COVID19_CAOE

Repository for COVID19 response models of CAOE team. 

## Overview

## Installation and how to run the code

- Download the code to your local path (e.g., .../COVID19-vaccine)

- Add both /COVID19-vaccine and /COVID19-vaccine/VaccineAllocation to your $PYTHONPATH

## Structure of the Code

### main_allocation.py

- The main module to run the simulation.

### policy_search_functions.py

- Run the policy search on training set, if a trigger policy is not given. 
Perform test simulation on the best found  policy.
- If a trigger policy is given perform test simulation for the given policy.

