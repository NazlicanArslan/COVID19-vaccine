import pickle
import os
from datetime import datetime as dt
import numpy as np
from VaccineAllocation import load_config_file,config_path
from reporting.plotting import plot_multi_tier_sims
from reporting.report_pdf import generate_report
from reporting.output_processors import build_report,build_report_tiers
import csv
import pdb
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # list all .p files from the output folder
    #folder_name = "Benchmarks"
    #fileList = os.listdir("output/{}".format(folder_name))
    fileList = os.listdir("output")
    for instance_raw in fileList:
        if ".p" in instance_raw:
            instance_name = instance_raw[:-2]
            #path_file = f'output/{folder_name}/{instance_name}.p'
            path_file = f'output/{instance_name}.p'
            with open(path_file, 'rb') as outfile:
                read_output = pickle.load(outfile)
            
            if "peak_ICU" in instance_raw:
                obj_type = "peak_ICU"
            elif "mean_ICU" in instance_raw:
                obj_type = "mean_ICU"
            elif "peak_IH" in instance_raw:
                obj_type = "peak_IH"
            elif "mean_IH" in instance_raw:
                obj_type = "mean_IH"
            elif "infected" in instance_raw:
                obj_type = "infected"
            elif "death" in instance_raw:
                obj_type = "death"  
                 
            instance, interventions_train, vaccines, stoch_replicas, sim_output, selected_vaccine_policy, obj_value, no_simulation_run, config, (crn_seeds, unique_seeds) = read_output
            list_of_objectives = []
            list_of_names = ['mean_death',                              
                             'mean_infected',
                             'peak_ICU',
                             'total_ICU_admit',
                             'peak_IHT',   
                             'total_IHT_admit',
                             'mean_death_whole_sim',                              
                             'mean_infected_whole_sim',
                             'peak_ICU_whole_sim',
                             'total_ICU_admit_whole_sim',
                             'peak_IHT_whole_sim',   
                             'total_IHT_admit_whole_sim']
            
            list_names_write = ['death',                              
                             'infected',
                             'peak ICU',
                             'total ICU',
                             'peak IHT',   
                             'total IHT',
                             'death whole sim',                              
                             'infected whole sim',
                             'peak ICU whole sim',
                             'total ICU whole sim',
                             'peak IHT whole sim',   
                             'total IHT whole sim']
            T = np.minimum(instance.T, instance.T)
            T0 = np.where(np.array(selected_vaccine_policy._instance.cal.calendar) == selected_vaccine_policy._vaccine_distribution_date[0])[0][0]

            for o_type in list_of_names:
                stoch = [np.round(rep_i['v_obj'][o_type],2) for rep_i in stoch_replicas]        
                expected = np.mean(stoch)
                list_of_objectives.append(expected)
            obj_type = 'mean_death'
            with open('results/results_rollout_' + instance_name + obj_type + '_'+ vaccines.vaccine_file_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(list_names_write)
                writer.writerow(list_of_objectives)
                writer.writerow([])
                writer.writerow(['','', 'total vaccine',	'planning horizon',	
                                	'A1-R1', "A1-R2", 'A2-R1',	'A2-R2', 'A3-R1', 
                                'A3-R2',	'A4-R1', 'A4-R2', 'A5-R1',	'A5-R2'])
                
                N = instance.N
                allocation = sum(sum(allocation_item['assignment'] for allocation_item in allocation_daily if allocation_item['which_dose'] == 1) 
                                 for allocation_daily in selected_vaccine_policy._allocation)
                allocation = np.round((allocation/N).reshape(1,10),2)
                total_vaccine = sum(sum(v) for v in vaccines.vaccine_proportion)
                horizon = len(vaccines.vaccine_proportion)
                output = np.concatenate(([''],[''],[total_vaccine], [horizon], allocation[0])) 
                writer.writerow(output)
                writer.writerow([])
                        
                writer.writerow(['Method','Date', 'Vaccine Type', 'vaccine dose type', 
                                  'A1-R1','	A1-R2','A2-R1','A2-R2','A3-R1','A3-R2', 
                                 'A4-R1','A4-R2','A5-R1','A5-R2']) 
                A = instance.A
                L = instance.L
                for ind, allocation_daily in enumerate(selected_vaccine_policy._allocation):
                    for allocation_item in allocation_daily:
                        if allocation_item['which_dose'] == 1:
                            prop = allocation_item['assignment'].reshape(1,10)                              
                            if ind == 0:
                                output = np.concatenate((['rollout'],[allocation_item['supply']['actual_time']], 
                                                     [allocation_item['supply']['type']], 
                                                     [allocation_item['supply']['dose_type']], 
                                                     np.round(prop[0],2), [np.round(obj_value)]))
                            else:
                                output = np.concatenate(([''],[allocation_item['supply']['actual_time']], 
                                                     [allocation_item['supply']['type']], 
                                                     [allocation_item['supply']['dose_type']], 
                                                     np.round(prop[0],2)))
                            writer.writerow(output)     
