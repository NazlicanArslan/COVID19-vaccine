import pickle
import numpy as np
import multiprocessing as mp
import datetime as dt
from collections import defaultdict
from interventions import create_intLevel, form_interventions
from itertools import product
from SEIYAHRD import simulate_p
from trigger_policies import MultiTierPolicy
from VaccineAllocation import config, logger, output_path
from utils import profile_log, print_profiling_log
from threshold_policy import run_multi_calendar, stochastic_iterator, policy_multi_iterator
from objective_functions import multi_tier_objective
from vaccine_policies import VaccineAllocationPolicy as VAP
from vaccine_policies import find_rollout_allocation, fix_hist_allocation
import iteround
datetime_formater = '%Y-%m-%d %H:%M:%S'
date_formater = '%Y-%m-%d'

def greedy_stochastic_allocation(instance,
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
                                 policy_field="ToIHT",
                                 policy_ub=None,
                                 dfo_obj=None,
                                 v_time_increment=None,
                                 risk_meas = None):
    
    # Set up for policy search: build interventions according to input tiers
    fixed_TR = list(filter(None, instance.cal.fixed_transmission_reduction))
    tier_TR = [item['transmission_reduction'] for item in tiers]
    uniquePS = sorted(np.unique(np.append(fixed_TR, np.unique(tier_TR))))
    sc_levels = np.unique([tier['school_closure'] for tier in tiers] + [0, 1])
    #
    #breakpoint()
    fixed_CO = list(filter(None, instance.cal.fixed_cocooning))
    co_levels = np.unique(np.append([tier['cocooning'] for tier in tiers], np.unique(fixed_CO)) + [0])
    intervention_levels = create_intLevel(sc_levels, co_levels, uniquePS)
    interventions_train = form_interventions(intervention_levels, instance.epi, instance.N)
 
    A, L, N = instance.A, instance.L, instance.N
    vac_assignment, eligible_population = fix_hist_allocation(vaccines, instance)
    N_total = np.sum(N)
    
    # Run the deterministic path without vaccine to be used as a benchmark
    args = {'vaccine_time': vaccines.vaccine_start_time,
            'vac_time_index' : 0,
            'previous_group' : [0,0],
            'active_group': [0,0],
            'eligible_population': eligible_population,
            'type':'deterministic'}
    selected_vaccine_policy = vaccine_policy
    # sim_configs = policy_multi_iterator(instance,
    #                                             tiers,
    #                                             vaccines, 
    #                                             obj_func,
    #                                             interventions_train,
    #                                             policy_class=policy_class,
    #                                             fixed_policy=policy,
    #                                             fixed_vaccine_policy=selected_vaccine_policy,
    #                                             policy_field=policy_field,
    #                                             policy_ub=policy_ub)
    # # Launch parallel simulation
    # all_outputs = simulate_p(mp_pool, sim_configs)
                    
    # # Get the simulation results
    # sim_output, cost, _trigger_policy, _vac_policy, seed_0, kwargs_out = all_outputs[0]
    #hosp_benchmark = [sim_output['IHT'][t].sum() for t in range(len(instance.cal.real_hosp))]
    hosp_benchmark = instance.real_hosp
    if vaccine_policy == None:
        print('no vaccine policy is given, find greedy allocation.')
        # This statement runs if no training seed is provided
        vac_assignment, eligible_population = fix_hist_allocation(vaccines, instance)
        if crn_seeds == []:
            # Generate seeds if it is not provided
            args = {'vaccine_time': vaccines.vaccine_start_time,
                    'vac_time_index' : 0,
                    'previous_group' : [0,0],
                    'active_group': [0,0],
                    'eligible_population': eligible_population,
                    'type':'stochastic'}
                
            # Run stochastic paths with given rollout_policy
            stoch_outputs_i = []  # List of valid samples if policy_i
            crn_seeds_i = []  # Common random number seed of policy_i (set once)
            total_train_reps = 0  # Total samples executed in the filtering procedure
            # =================================================
            # Sample until required number of replicas achieved
            n_loops = 0
            while len(stoch_outputs_i) < n_replicas_train:
                chunksize = 1 if mp_pool is None else mp_pool._processes
                chunksize = chunksize if crn_seeds == [] else n_replicas_train
                total_train_reps += chunksize
                n_loops += chunksize + 1 if crn_seeds == [] else 0
                if crn_seeds == []:
                    # no input seeds
                    seed_shift_var=n_loops
                    crn_input = None
                    chunkinput = chunksize
                else:
                  seed_shift_var = 0
                  crn_input = crn_seeds[total_train_reps-chunksize:total_train_reps]
                  if len(crn_input) == 0:
                      # if the given seeds are run out, need to generate new seeds
                      crn_input = None
                      chunkinput = chunksize
                      seed_shift_var = crn_seeds[-1] + 1 + total_train_reps
                  else:
                      chunkinput = len(crn_input)
                 
                # Simulate n=chunksize samples of policy_i
                out_sample_configs = stochastic_iterator(instance,
                                                          tiers,
                                                          vaccines,
                                                          obj_func,
                                                          interventions_train,
                                                          policy_class=policy_class,
                                                          fixed_policy=policy,
                                                          fixed_vaccine_policy=selected_vaccine_policy,
                                                          policy_field=policy_field,
                                                          policy_ub=policy_ub,
                                                          seed_shift=seed_shift_var,
                                                          crn_seeds=crn_input,
                                                          n_replicas=chunkinput)
                                   
                out_sample_outputs = simulate_p(mp_pool, out_sample_configs)      
    
                for sample_ij in out_sample_outputs:
                    sim_j, cost_j, policy_j, _vac_policy, seed_j, kwargs_j = sample_ij
                    
                    real_hosp_end_ix = len(hosp_benchmark) 
                    IH_sim = sim_j['IHT'][0:real_hosp_end_ix]
                    IH_sim = IH_sim.sum(axis=(2,1))
                    f_benchmark = hosp_benchmark
                    
                    rsq = 1 - np.sum(((np.array(IH_sim) - np.array(f_benchmark))**2))/sum((np.array(f_benchmark) - np.mean(np.array(f_benchmark)))**2)
                    if rsq > 0.75:
                        stoch_outputs_i.append(sample_ij)
                        crn_seeds_i.append(seed_j)
                        if len(stoch_outputs_i) == n_replicas_train:
                            break
              
            # Save CRN seeds for all policies yet to be evaluated
            if crn_seeds == []:
                assert len(np.unique(crn_seeds_i)) == n_replicas_train
                crn_seeds = crn_seeds_i.copy()
                                 
    
        # Given training seeds, obtain the best vaccine policy
        vac_assignment, eligible_population = fix_hist_allocation(vaccines, instance)
        print('training seeds are ready, obtain the best vaccine policy ')        
        args = {'vaccine_time': vaccines.vaccine_start_time,
                'vac_time_index' : 0,
                'previous_group' : [0,0],
                'active_group': [0,0],
                'eligible_population': eligible_population,
                'type':'stochastic'}
        candidate_group = [0, 0]
        no_simulation_run = 0    
        end_vaccine_time = vaccines.vaccine_time[len(vaccines.vaccine_time)-1]  
        while args['vaccine_time'] <= end_vaccine_time and sum(sum(args['eligible_population'])) > 100:
            
            v_time = args['vaccine_time'] 
            print(v_time)
            best_obj_value_per_vaccine = float("inf")
            
            for age in range(2, 5):
                for risk in range(L):
                    temp_assignment = vac_assignment.copy()
             
                    total_daily_supply = 0
                    for supply in vaccines.vaccine_supply:
                        for s_item in supply:
                            if s_item['time'] == v_time:
                                total_daily_supply += s_item['amount'] * N_total
                                
                    if args['eligible_population'][age][risk] > 0 and \
                        args['eligible_population'][age][risk] > total_daily_supply: 
                        args['active_group'] = [age, risk]
                        # Construct the policy to compare the allocation in vaccine time.
                        # This is not the actual policy.
                        rollout_policy = VAP.vaccine_rollout_policy(instance, 
                                                                    vaccines, 
                                                                    temp_assignment,
                                                                    args)
    
                        no_simulation_run += 1
                        
                        # Run stochastic paths with given rollout_policy
                        stoch_outputs_i = []  # List of valid samples if policy_i
                        crn_seeds_i = []  # Common random number seed of policy_i (set once)
                        total_train_reps = 0  # Total samples executed in the filtering procedure
                        # =================================================
                        # Sample until required number of replicas achieved
                        n_loops = 0
                        while len(stoch_outputs_i) < n_replicas_train:
                            chunksize = 1 if mp_pool is None else mp_pool._processes
                            chunksize = chunksize if crn_seeds == [] else n_replicas_train
                            total_train_reps += chunksize
                            n_loops += chunksize + 1 if crn_seeds == [] else 0
                            seed_shift_var = 0
                            crn_input = crn_seeds[total_train_reps-chunksize:total_train_reps]
                            if len(crn_input) == 0:
                                # if the given seeds are run out, need to generate new seeds
                                crn_input = None
                                chunkinput = chunksize
                                seed_shift_var = crn_seeds[-1] + 1 + total_train_reps
                            else:
                                chunkinput = len(crn_input)
                             
                            # Simulate n=chunksize samples of policy_i
                            out_sample_configs = stochastic_iterator(instance,
                                                                      tiers,
                                                                      vaccines,
                                                                      obj_func,
                                                                      interventions_train,
                                                                      policy_class=policy_class,
                                                                      fixed_policy=policy,
                                                                      fixed_vaccine_policy=rollout_policy,
                                                                      policy_field=policy_field,
                                                                      policy_ub=policy_ub,
                                                                      seed_shift=seed_shift_var,
                                                                      crn_seeds=crn_input,
                                                                      n_replicas=chunkinput)
                                               
                            out_sample_outputs = simulate_p(mp_pool, out_sample_configs)      
    
                            for sample_ij in out_sample_outputs:
                              sim_j, cost_j, policy_j, _vac_policy, seed_j, kwargs_j = sample_ij
                              stoch_outputs_i.append(sample_ij)
                              
                        stoch_replicas = [rep_i[0] for rep_i in stoch_outputs_i]
                        stoch_costs = [rep_i['v_obj'][dfo_obj] for rep_i in stoch_replicas]        

                        if risk_meas == "expected_value":
                            cost = np.mean(stoch_costs)
                        elif risk_meas == "cVaR":
                            p = 1/len(stoch_costs)
                            lamb = 0.5
                            var_level = 0.9
                            var_value = np.percentile(stoch_costs, var_level*100)
                            cost = lamb*(1/(1 - var_level))*sum(stoch_c*p for stoch_c in stoch_costs if stoch_c >= var_value ) + (1 - lamb)*np.mean(stoch_costs)
                    
                    
                        # Check objective value per vaccine:
                        vaccine_used = 0
                        for allocation_daily in rollout_policy._allocation:
                            vaccine_used += sum(sum(sum(allocation_item['assignment'])) for allocation_item in allocation_daily)
                        
                        if cost/vaccine_used < best_obj_value_per_vaccine:
                            best_obj_value_per_vaccine = cost/vaccine_used
                            best_obj_value = cost
                            candidate_group = [age, risk]
                            
            # The group end time of previous group will be the start date of the next group.
            vac_assignment, args['vaccine_time'], args['eligible_population'] = find_rollout_allocation(instance,vaccines, v_time_increment,
                                                                                                                               temp_assignment, 
                                                                                                                               candidate_group, 
                                                                                                                               args)        
            args['previous_group'] = candidate_group
    
        selected_vaccine_policy = VAP.vaccine_policy(instance, vaccines, vac_assignment, args['type'])
    else:
        selected_vaccine_policy = vaccine_policy

    # Run stochastic test paths with final rollout_policy
    vac_assignment, eligible_population = fix_hist_allocation(vaccines, instance)
    args = {'vaccine_time': vaccines.vaccine_start_time,
            'vac_time_index' : 0,
            'previous_group' : [0,0],
            'active_group': [0,0],
            'eligible_population': eligible_population,
            'type':'stochastic'}
    stoch_outputs_test = []  # List of valid samples if policy_i
    unique_seeds = []  # Common random number seed of policy_i (set once)
    total_test_reps = 0  # Total samples executed in the filtering procedure
    stoch_outputs_test = []
    # =================================================
    # Sample until required number of replicas achieved
    print('testing')
    while len(stoch_outputs_test) < n_replicas_test:
        chunksize = 4 if mp_pool is None else mp_pool._processes
        total_test_reps += chunksize

        if unique_seeds_ori == []:
            # no input seeds
            seed_shift_var = 10_00000 + total_test_reps
            crn_input = None
            chunkinput = chunksize
        else:
          seed_shift_var = 0
          crn_input = unique_seeds_ori[total_test_reps-chunksize:total_test_reps]
          if len(crn_input) == 0:
              # if the given seeds are run out, need to generate new seeds
              crn_input = None
              chunkinput = chunksize
              seed_shift_var = unique_seeds_ori[-1] + 1 + total_test_reps
          else:
              chunkinput = len(crn_input)
         
        # Simulate n=chunksize samples of policy_i
        out_sample_configs = stochastic_iterator(instance,
                                                  tiers,
                                                  vaccines,
                                                  obj_func,
                                                  interventions_train,
                                                  policy_class=policy_class,
                                                  fixed_policy=policy,
                                                  fixed_vaccine_policy=selected_vaccine_policy,
                                                  policy_field=policy_field,
                                                  policy_ub=policy_ub,
                                                  seed_shift=seed_shift_var,
                                                  crn_seeds=crn_input,
                                                  n_replicas=chunkinput)
                           
        out_sample_outputs = simulate_p(mp_pool, out_sample_configs)      
        #breakpoint()
        for sample_ij in out_sample_outputs:
            sim_j, cost_j, policy_j, _vac_policy, seed_j, kwargs_j = sample_ij
            real_hosp_end_ix = len(hosp_benchmark) 
            IH_sim = sim_j['IHT'][0:real_hosp_end_ix]
            IH_sim = IH_sim.sum(axis=(2,1))
            f_benchmark = hosp_benchmark
          
            rsq = 1 - np.sum(((np.array(IH_sim) - np.array(f_benchmark))**2))/sum((np.array(f_benchmark) - np.mean(np.array(f_benchmark)))**2)
            print('rsq ', rsq)
            if rsq > 0.75:   
                #breakpoint()
                stoch_outputs_test.append(sample_ij)
                unique_seeds.append(seed_j)
            if len(stoch_outputs_test) == n_replicas_test:
                break
        
            
    assert len(np.unique(unique_seeds)) == n_replicas_test          
    stoch_replicas = [rep_i[0] for rep_i in stoch_outputs_test]
    stoch_costs = [rep_i['v_obj'][dfo_obj] for rep_i in stoch_replicas]        
    expected_cost = np.mean(stoch_costs)    
    #breakpoint()
    
    instance_name = instance_name if instance_name is not None else f'output_{instance.city}.p'
    file_path = output_path / f'{instance_name}.p'
    if file_path.is_file():
        file_path = output_path / f'{instance_name}_{str(dt.datetime.now())}.p'    
    
    sim_output = []
    no_simulation_run = 0
    selected_vaccine_policy = []
    with open(str(file_path), 'wb') as outfile:
        pickle.dump(
            (instance, interventions_train, kwargs_j, policy_j, vaccines, stoch_replicas, sim_output,
             selected_vaccine_policy, expected_cost, no_simulation_run, config,
             (crn_seeds, unique_seeds)),
            outfile, pickle.HIGHEST_PROTOCOL)
    

        
        
    return selected_vaccine_policy, expected_cost 
