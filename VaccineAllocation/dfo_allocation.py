import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
import datetime as dt
from collections import defaultdict
import multiprocessing as mp
from utils import parse_arguments
from VaccineAllocation import load_config_file, logger, change_paths
from pathlib import Path
from plotting_vaccine import read_hosp

import numpy as np
import multiprocessing as mp
import datetime as dt
from collections import defaultdict
from interventions import create_intLevel, form_interventions
from itertools import product
from SEIYAHRD import simulate_p
from vaccine_policies import build_vaccine_policy_candidates, MultiTierPolicy
from VaccineAllocation import config, logger, output_path
from utils import profile_log, print_profiling_log
from threshold_policy import run_multi_calendar, stoch_simulation_iterator, vaccine_policy_multi_iterator
from objective_functions import multi_tier_objective
from vaccine_policies import MultiTierPolicy as MTP
from vaccine_policies import VaccineAllocationPolicy as VAP

#from scipy.optimize import least_squares
#from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES
#from scipy.optimize._lsq.common import EPS, make_strictly_feasible
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
#import nlopt
import math 

def deterministic_vaccine_path(instance,
                                 tiers,
                                 vaccines,
                                 obj_func,
                                 n_replicas_train=100,
                                 n_replicas_test=100,
                                 instance_name=None,
                                 policy_class='constant',
                                 policy=None,
                                 vaccine_policy=None,
                                 mp_pool=None,
                                 crn_seeds=[],
                                 unique_seeds_ori=[],
                                 forcedOut_tiers=None,
                                 redLimit=1000,
                                 after_tiers=[0,1,2,3,4],
                                 policy_field="IYIH",
                                 policy_ub=None):
    '''
        TODO: Write proper docs
    '''
    
    #    Set up for policy search: build interventions according to input tiers
    fixed_TR = list(filter(None, instance.cal.fixed_transmission_reduction))
    tier_TR = [item['transmission_reduction'] for item in tiers]
    uniquePS = sorted(np.unique(np.append(fixed_TR, np.unique(tier_TR))))
    sc_levels = np.unique([tier['school_closure'] for tier in tiers] + [0, 1])
    fixed_CO = list(filter(None, instance.cal.fixed_cocooning))
    co_levels = np.unique(np.append([tier['cocooning'] for tier in tiers], np.unique(fixed_CO)) + [0])
    intervention_levels = create_intLevel(sc_levels, co_levels, uniquePS)
    interventions_train = form_interventions(intervention_levels, instance.epi, instance.N)
    t_start = 10#instance.epi.t_start
    # Build an iterator of all the candidates to be simulated by simulate_p
    sim_configs = vaccine_policy_multi_iterator(instance,
                                                tiers,
                                                vaccines, 
                                                obj_func,
                                                interventions_train,
                                                policy_class=policy_class,
                                                fixed_policy=policy,
                                                fixed_vaccine_policy=vaccine_policy,
                                                policy_field=policy_field,
                                                policy_ub=policy_ub)
    # Launch parallel simulation
    all_outputs = simulate_p(mp_pool, sim_configs)
    
    # Search of the best feasible candidate
    best_cost, best_sim, best_policy, best_vac_policy, best_params = np.Inf, None, None, None, None
    hosp_benchmark = None
    if len(all_outputs) == 1:
        # Skip search if there is only one candidate
        sim_output, cost, best_policy, best_vac_policy, seed_0, kwargs_out = all_outputs[0]

    return sim_output

def compute_objective(x_age_risk, kwargs):

    instance = kwargs['instance']
    vaccines = kwargs['vaccines']
    tiers = kwargs['tiers']

    # TODO Read command line args for n_proc for better integration with crunch
    n_proc = 1
    
    # TODO: pull out n_replicas_train and n_replicas_test to a config file
    n_replicas_train = 1
    n_replicas_test = 1
    
    # Create the pool (Note: pool needs to be created only once to run on a cluster)
    mp_pool = mp.Pool(n_proc) if n_proc > 1 else None
    
    # check if the "do-nothing" / 'Stage 1 option is in the tiers. If not, add it
    originInt = {
        "name": "Stage 1",
        "transmission_reduction": 0,
        "cocooning": 0,
        "school_closure": 0,
        "min_enforcing_time": 1,
        "daily_cost": 0,
        "color": 'white'
        }
   
    if tiers.tier_type == 'constant':
        originInt["candidate_thresholds"] = [-1]  # Means that there is no lower bound
    elif tiers.tier_type == 'step':
        originInt["candidate_thresholds"] = [[-1], [-0.5]]
    
    if not (originInt in tiers.tier):
        tiers.tier.insert(0, originInt)
    
    #Fix some thresholds, the numbers are not important
    given_threshold = eval('[-1,0,5,20,70]')
    given_date = None
    # if a threshold/threshold+stepping date is given, then it carries out a specific task
    # if not, then search for a policy
    selected_policy = None
    if tiers.tier_type == 'constant':
        if given_threshold is not None:
            selected_policy = MTP.constant_policy(instance, tiers.tier, given_threshold)
    elif tiers.tier_type == 'step':
        if (given_threshold is not None) and (given_date is not None):
            selected_policy = MTP.step_policy(instance, tiers.tier, given_threshold, given_date)
    
    #Update the allocation decision variables
    x_age_risk = x_age_risk.reshape(5,2)
    allocation = []
    for id_t, supply in enumerate(vaccines.vaccine_supply):
        allocation_list = []
        for id_v, s_item in enumerate(supply):
            allocation_item = {'proportion': x_age_risk, 'supply': s_item, 'which_dose': 1}    
            allocation_list.append(allocation_item)
        allocation.append(allocation_list)
    selected_vaccine_policy = VAP.vaccine_policy(instance, vaccines, allocation)
    #print(selected_vaccine_policy._vaccine_age_risk_pro)
    task_str = str(selected_policy) if selected_policy is not None else f'opt{len(tiers.tier)}'
    instance_name = 'det_path'
    # read in the policy upper bound
    policy_ub = None
    sim_output = deterministic_vaccine_path(instance=instance,
                                            tiers=tiers.tier,
                                            vaccines = vaccines,
                                            obj_func=multi_tier_objective,
                                            n_replicas_train=n_replicas_train,
                                            n_replicas_test=n_replicas_test,
                                            instance_name=instance_name,
                                            policy_class=tiers.tier_type,
                                            policy=selected_policy,
                                            vaccine_policy=selected_vaccine_policy,
                                            mp_pool=None,
                                            crn_seeds=[],
                                            unique_seeds_ori=[],
                                            forcedOut_tiers=eval('[]'),
                                            redLimit=100000,
                                            after_tiers=eval('[0,1,2,3,4]'),
                                            policy_field='IYIH',
                                            policy_ub=policy_ub)
    
    obj_values = sim_output['v_obj']    
    return obj_values['mean_death'], obj_values['mean_infected'], obj_values['peak_ICU'], obj_values['peak_IH'], obj_values['mean_ICU'], obj_values['mean_IH']

def objective(x, kwargs):
    death, infected, peak_ICU, peak_IH, mean_ICU, mean_IH = compute_objective(x, kwargs)
    which_obj = kwargs['obj']
    
    if which_obj == 'death':
        print(death)
        return death
    elif which_obj == 'infected':
        print(infected)
        return infected
    elif which_obj == 'peak_ICU':
        print(peak_ICU)
        return peak_ICU
    elif which_obj == 'peak_IH':
        print(peak_IH)
        return peak_IH
    elif which_obj == 'mean_ICU':
        print(mean_ICU)
        return mean_ICU
    elif which_obj == 'mean_IH':
        print(mean_IH)
        return mean_IH
    

def cobyla_fit(initial_guess, kwargs):
    bounds=((0, 1),(0, 1),(0, 1),(0, 1),(0, 1),(0, 1),(0, 1),(0, 1),(0, 1),(0, 1))
    
    #Construct the bounds in the form of constraints
    cons = []
    for factor in range(len(bounds)):
        lower, upper = bounds[factor]
        l = {'type': 'ineq', 
             'fun': lambda x, lb=lower, i=factor: x[i] - lb}
        u = {'type': 'ineq', 
             'fun': lambda x, ub=upper, i=factor: ub - x[i]}
        cons.append(l)
        cons.append(u)
     
    #Sum of proportions should be less than 1
    sum_x = {'type': 'ineq', 
             'fun': lambda x: 1 - x[0] - x[1] - x[2] - x[3] - x[4] - x[5] - x[6] - x[7] - x[8] - x[9]}
    cons.append(sum_x)
    
    #Assigned vaccines should be less than the population of each age-risk group
    instance = kwargs['instance']
    vaccines = kwargs['vaccines']
    N = np.reshape(instance.N, (10))
    available_vaccine = np.round(vaccines.vaccine_proportion[0][0] * np.sum(instance.N))
    print('available vaccine', available_vaccine )
    for factor in range(len(bounds)):
        upper = N[factor]
        u = {'type': 'ineq', 
             'fun': lambda x, ub=upper, i=factor: ub - x[i]*available_vaccine}
        cons.append(u)
        
    result = minimize(objective, initial_guess, method='COBYLA', constraints=cons, tol = 1e-6, args = kwargs, options={'rhobeg': 0.01})
    return result
 
def find_best_initial_solution(instance, tiers, vaccines, obj_func, n_replicas_train=100, n_replicas_test=100, 
                               instance_name=None, policy_class='constant', policy=None, vaccine_policy=None,
                               mp_pool=None, crn_seeds=[], unique_seeds_ori=[], forcedOut_tiers=None,
                               redLimit=1000, after_tiers=[0,1,2,3,4], policy_field="IYIH", policy_ub=None, dfo_obj=None, initial=0):
    initial = 0
    n_proc = 1
    n_replicas_train = 1
    n_replicas_test = 1
    mp_pool = mp.Pool(n_proc) if n_proc > 1 else None
    originInt = {
        "name": "Stage 1",
        "transmission_reduction": 0,
        "cocooning": 0,
        "school_closure": 0,
        "min_enforcing_time": 1,
        "daily_cost": 0,
        "color": 'white'
        }
    if tiers.tier_type == 'constant':
        originInt["candidate_thresholds"] = [-1]  # Means that there is no lower bound
    elif tiers.tier_type == 'step':
        originInt["candidate_thresholds"] = [[-1], [-0.5]]
    if not (originInt in tiers.tier):
        tiers.tier.insert(0, originInt)
    #Fix some thresholds, the numbers are not important
    given_threshold = eval('[-1,0,5,20,70]')
    given_date = None
    selected_policy = None
    if tiers.tier_type == 'constant':
        if given_threshold is not None:
            selected_policy = MTP.constant_policy(instance, tiers.tier, given_threshold)
    elif tiers.tier_type == 'step':
        if (given_threshold is not None) and (given_date is not None):
            selected_policy = MTP.step_policy(instance, tiers.tier, given_threshold, given_date)
   
    if initial == 0:
        selected_vaccine_policies = [VAP.high_risk_senior_first_vaccine_policy(instance, vaccines),
                                     VAP.senior_first_vaccine_policy(instance, vaccines),
                                     VAP.min_death_vaccine_policy(instance, vaccines),
                                     VAP.min_infected_vaccine_policy(instance, vaccines),
                                     VAP.sort_contact_matrix_vaccine_policy(instance, vaccines),
                                     VAP.low_risk_young_first_vaccine_policy(instance, vaccines)]
    elif initial == 1:
        selected_vaccine_policies = [VAP.high_risk_senior_first_vaccine_policy(instance, vaccines)]
    elif initial == 2:
        selected_vaccine_policies = [VAP.senior_first_vaccine_policy(instance, vaccines)]
    elif initial == 3:
        selected_vaccine_policies = [VAP.min_death_vaccine_policy(instance, vaccines)]
    elif initial == 4:
        selected_vaccine_policies = [VAP.min_infected_vaccine_policy(instance, vaccines)]
    elif initial == 5:
        selected_vaccine_policies = [VAP.sort_contact_matrix_vaccine_policy(instance, vaccines)]
    elif initial == 6:
        selected_vaccine_policies = [VAP.low_risk_young_first_vaccine_policy(instance, vaccines)]
    elif initial == 7:
        selected_vaccine_policies = [VAP.proportional_to_pop(instance, vaccines)]
    
    best_obj = float("inf")
    list_of_initial_policies = []
    for v_policy in selected_vaccine_policies:
        print(v_policy._vaccine_policy_name)
                
        sim_output = deterministic_vaccine_path(instance=instance,
                                                 tiers=tiers.tier,
                                                 vaccines = vaccines,
                                                 obj_func=multi_tier_objective,
                                                 n_replicas_train=n_replicas_train,
                                                 n_replicas_test=n_replicas_test,
                                                 instance_name=instance_name,
                                                 policy_class=tiers.tier_type,
                                                 policy=selected_policy,
                                                 vaccine_policy=v_policy,
                                                 mp_pool=None,
                                                 crn_seeds=[],
                                                 unique_seeds_ori=[],
                                                 forcedOut_tiers=eval('[]'),
                                                 redLimit=100000,
                                                 after_tiers=eval('[0,1,2,3,4]'),
                                                 policy_field='IYIH',
                                                 policy_ub=policy_ub)
        
        obj_values = sim_output['v_obj']


        print('death:', np.round(obj_values['mean_death'], 1), 'ICU:', np.round(obj_values['peak_ICU'], 1), 'IH:', np.round(obj_values['peak_IH'], 1), 
              'infected:', np.round(obj_values['mean_infected'], 1), 'ICU:', np.round(obj_values['mean_ICU'], 1))
        
        if dfo_obj == 'death':
            obj_value = obj_values['mean_death']
        elif dfo_obj == 'peak_ICU':
            obj_value = obj_values['peak_ICU']
        elif dfo_obj == 'peak_IH':
            obj_value = obj_values['peak_IH']
        elif dfo_obj == 'infected':
            obj_value = obj_values['mean_infected']
        elif dfo_obj == 'mean_ICU':
            obj_value = obj_values['mean_ICU']
        elif dfo_obj == 'mean_IH':
            obj_value = obj_values['mean_IH']
            
        policy_temp ={'v_policy': v_policy, 'obj_value': obj_value} 
        list_of_initial_policies.append(policy_temp)
        
        print('obj', obj_value)
        if obj_value < best_obj:
            best_policy = v_policy#.copy()#None # 1*v_policy._vaccine_age_risk_pro
            best_obj = 1*obj_value
            
    return best_policy, best_obj, list_of_initial_policies

def run_allocation(instance,
                   tiers,
                   vaccines,
                   obj_func,
                   n_replicas_train=100,
                   n_replicas_test=100,
                   instance_name=None,
                   policy_class='constant',
                   policy=None,
                   vaccine_policy=None,
                   mp_pool=None,
                   crn_seeds=[],
                   unique_seeds_ori=[],
                   forcedOut_tiers=None,
                   redLimit=1000,
                   after_tiers=[0,1,2,3,4],
                   policy_field="IYIH",
                   policy_ub=None,
                   method="COBYLA",
                   dfo_obj=None,
                   initial=None):
    

    # initial guess
    # pick the best initial guess
    print(method)
    best_initial_policy, best_initial_obj, list_of_initial_policies = find_best_initial_solution(instance, tiers, vaccines, obj_func, n_replicas_train=1, n_replicas_test=1, 
                                                                           instance_name=instance_name, policy_class='constant', policy=policy, vaccine_policy=vaccine_policy,
                                                                           mp_pool=mp_pool, crn_seeds=crn_seeds, unique_seeds_ori=unique_seeds_ori, forcedOut_tiers=forcedOut_tiers,
                                                                           redLimit=redLimit, after_tiers=[0,1,2,3,4], policy_field="IYIH", policy_ub=policy_ub, dfo_obj=dfo_obj)

    print('best initial allocation:', best_initial_obj) 
    
    for id_date, allocation_daily in enumerate(best_initial_policy._allocation):
        for allocation_item in allocation_daily:
            x = allocation_item['proportion'].reshape(10)
                       

    #these will be used by other functions
    kwargs  = {'instance' : instance,
               'vaccines' : vaccines,
               'tiers' : tiers,
               'obj' : dfo_obj
               }
    ########## ########## ##########
    #Modify this part for different methods
    res = cobyla_fit(x, kwargs)
    ########## ########## ##########  
    #Get variable value
    obj_value = res.fun
    x_age_risk = res.x
    
    ##Update the allocation decision variables
    x_age_risk = x_age_risk.reshape(5,2)
    allocation = []
    for id_t, supply in enumerate(vaccines.vaccine_supply):
        allocation_list = []
        for id_v, s_item in enumerate(supply):
            allocation_item = {'proportion': x_age_risk, 'supply': s_item, 'which_dose': 1}    
            allocation_list.append(allocation_item)
        allocation.append(allocation_list)
    selected_vaccine_policy = VAP.vaccine_policy(instance, vaccines, allocation)
    
    instance_name = instance_name if instance_name is not None else f'output_{instance.city}.p'
    file_path = output_path / f'{instance_name}.p'
    if file_path.is_file():
        file_path = output_path / f'{instance_name}_{str(dt.datetime.now())}.p'    
    
    with open(str(file_path), 'wb') as outfile:
        pickle.dump(
            (instance, selected_vaccine_policy, obj_value, config),
            outfile, pickle.HIGHEST_PROTOCOL)
        
    return selected_vaccine_policy, obj_value
 