'''
This module includes different vaccine allocation policies that are simulated
'''
import json
import numpy as np
from VaccineAllocation import config
from itertools import product, permutations
import copy 
import iteround
import datetime as dt
from trigger_policies import MultiTierPolicy

def find_allocation(vaccines, calendar, A, L, T, N, allocation):
    
    for allocation_type in allocation:
        for allocation_item in allocation[allocation_type]:
            id_t = np.where(calendar == allocation_item['supply']['time'])[0][0]
            vaccine_a = []
            for i in range(A):
                for j in range(L):
                    vaccine_a.append([0] * (T))
                         
            k = 0    
            for i in range(A):
                for j in range(L):
                    vaccine_a[k][id_t] = allocation_item['assignment'][i, j] 
                    k += 1   
            allocation_item['assignment_daily'] = np.array(vaccine_a)
    
    return allocation
            
class VaccineAllocationPolicy():
    '''
        A vaccine allocation policy for different age-risk groups.

    '''
    def __init__(self, instance, vaccines, allocation, problem_type, policy_name):
        # Initialize
        self._instance = instance
        self._vaccines = vaccines
        self._vaccine_groups = vaccines.define_groups(problem_type)
        T, A, L = instance.T, instance.A, instance.L
        total_risk_gr = A*L
        N = instance.N
                    
        #self._vaccine_distribution_date = np.unique(vaccines.vaccine_time)
        self._vaccine_policy_name = policy_name
        self._allocation = allocation

        # Checking the feasibility
        print('Considering', self._vaccine_policy_name)
        total_allocation_p_a_r = np.zeros((A, L))
        total_allocation = np.zeros((A, L))
        for allocation_item in allocation['v_first']:
            total_allocation_p_a_r += allocation_item['within_proportion']
            total_allocation += allocation_item['assignment']
        
        
        print('Total allocated vaccine first dose (within-proportion):')
        print(np.round(total_allocation_p_a_r, 4))
        
        total_allocation_p_a_r = np.zeros((A, L))
        total_allocation = np.zeros((A, L))
        for allocation_item in allocation['v_wane']:
            total_allocation_p_a_r += allocation_item['within_proportion']
            total_allocation += allocation_item['assignment']
        
        
        print('Total allocated waneddose (within-proportion):')
        print(np.round(total_allocation_p_a_r, 4))
        
        total_allocation_p_a_r = np.zeros((A, L))
        total_allocation = np.zeros((A, L))
        for allocation_item in allocation['v_booster']:
            total_allocation_p_a_r += allocation_item['within_proportion']
            total_allocation += allocation_item['assignment']
        
        
        print('Total allocated booster dose (within-proportion):')
        print(np.round(total_allocation_p_a_r, 4))
        
        # Assigning from-to vaccine groups for the SEIR model
        for v_group in self._vaccine_groups:
            v_group.vaccine_flow(instance, allocation)
        
   
    def reset_vaccine_history(self, instance, seed):
        '''
            reset vaccine history for a new simulation.
        '''
        for v_group in self._vaccine_groups:        
            v_group.reset_history(instance, seed)
        
        
    @classmethod
    def vaccine_policy(cls, instance, vaccines, problem_type):
        T, A, L = instance.T, instance.A, instance.L
        N = instance.N
        N_total = np.sum(N)
        calendar = np.array(instance.cal.calendar)
        allocation = vaccines.vaccine_allocation
        #breakpoint()
        allocation = find_allocation(vaccines, calendar, A, L, T, N, allocation)
        #breakpoint()
        return cls(instance, vaccines, allocation, problem_type, 'fixed_policy')
    

def fix_hist_allocation(vaccines, instance):
    A, L, N = instance.A, instance.L, instance.N
    uptake = 1
    eligible_population = N.copy()*uptake
    eligible_population[0][0] = 0
    eligible_population[0][1] = 0
    eligible_population[1][0] = 0
    eligible_population[1][1] = 0
        
    allocation = []
    # Fixed the historical allocations.
    if vaccines.vaccine_allocation_file_name is not None: 
        for allocation_daily in vaccines.fixed_vaccine_allocation:
            for allocation_item in allocation_daily:
                vac_assignment = allocation_item['assignment']
                allocation_item['within_proportion'] = vac_assignment/N
                if allocation_item['which_dose'] == 1:
                    for age in range(A):
                        for risk in range(L):
                            eligible_population[age][risk] -= vac_assignment[age][risk]
                                       
        allocation = vaccines.fixed_vaccine_allocation.copy() 
    else:
        allocation = []
        
    #breakpoint()    
    return allocation, eligible_population

