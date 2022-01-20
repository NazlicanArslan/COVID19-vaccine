# COVID-19

Repository is part of COVID-19 Vaccine Planning Tool of Rockefeller Foundation Project

## Overview

This software is vaccine and variant of concern incorporated version of the following repository: https://github.com/haoxiangyang89/COVID_Staged_Alert

## Installation and how to run the code

- Download and unzip the code to a local path (e.g., .../COVID19-vaccine-main/VaccineAllocation)

- The following packages are required:
  - matplotlib
  - pandas
  - numpy
  - Scipy

- Add both /COVID19-vaccine and /COVID19-vaccine/VaccineAllocation to your $PYTHONPATH

## Running the Code
- Type the following code into Python Command Window before running any code 
  -Import sys 
  -sys.path.append(.../COVID19-vaccine-main)
  -sys.path.append(.../COVID19-vaccine-main/VaccineAllocation)

## Structure of the Code

## Config Directory 
- Contains configuration files necessary to initialize running a sample path

## Data Processing directory 
- Contains death data, hospitalization, code cleaning, and seed generation files 

## Instances directory
- Contains related input .csv and .json files such as Omicron prevalence and hospitalization data 

## Output directory
- Used to store output files from Crunch. These files will be used in seed generation or plot generation

## main_allocation.py
- The main module to run the simulation.
- Running a sample path in main_allocation.py
  -  runfile('... /COVID19-vaccine-main/VaccineAllocation/main_allocation.py', 'austin -f setup_data_Final.json -t tiers5_opt_Final.json -train_reps 0 -test_reps 1 -f_config austin_test_IHT.json -n_proc 1 -tr transmission_new.csv -hos austin_real_hosp_updated.csv  -v_allocation vaccine_allocation_fixed.csv  -seed new_seed_Nov.p -n_policy 7  -v_boost=booster_allocation_fixed_50.csv -gt [-1,5,15,30,50]', wdir='.../COVID19-vaccine-main/VaccineAllocation')


## Pipelinemultitier.py
- Responsible for generating plots after output from Crunch

## policy_search_functions.py

- Run the policy search on training set, if a trigger policy is not given. 
Perform test simulation on the best found policy.
- If a trigger policy is given perform test simulation for the given policy.

## Epi_params.py
- Contains epidemiological parameters from the TACC model

## Interventions.py
- Defines the knobs of an interventions and forms the available interventions considering school closures, cocooning, and different levels of social distancing

## SEIYAHRD.py
- Simulate the SEIR model with vaccines included, considering different age groups and seven compartments

## Trigger_policies.py
- Different trigger policies that are simulated

## Utils.py
- Timing and rounding functions

## Vaccine_params.py
- Defines epidemiological characteristics and includes supply and fixed allocation schedule of vaccine 

## Vaccine_policies.py
- Includes different vaccine allocation policies that are simulated 

## Least_squares_fit.py
-  Minimizes a weighted sum of least-square errors to fit transmission-reduction parameters and certain dynamics in use of the ICU and hospital duration

## Init.py

## ACS_script.py

## main_ACS.py

## Objective_functions.py

## Threshold_policy.py


## Crunch 

## Copying to Crunch 

- Use the following command 
  - rsync -av --max-size=10m --exclude '*.pdf' --exclude '*.aux' --exclude '*.log' --exclude '*.tex' --exclude '*.aux' --exclude '*.p' --exclude '*.pyc' --exclude 'Old' --exclude 'presentations' VaccineAllocation/ [username]@crunch.osl.northwestern.edu:scratch
- Ensure that the VaccineAllocation directory is located in Scratch
- The vaccine_batch.sh file should be in Scratch, not VaccineAllocation directory 

## Running on Crunch
- Use the following command
  - cd scratch
  - qsub vaccine_batch.sh



- Use the following command to view submission status or delete submission
  - qdel: delete submission
  - qstat -u “*”: check submission status 

- Option to change number of test or train reps through test_rep, train_rep

- Change number of cores by changing n_proc and -pe smp 

## Downloading Output
- The .p file (data file) will be generated in /output 
- After the submission is run, save the .p file from /scratch/VaccineAllocation/output by right-clicking and download into your own computer (.../COVID19-vaccine-main/VaccineAllocation/output)
- Generate plots using pipelinemultitier.py or generate seeds using seeds_read.py in the data processing directory
