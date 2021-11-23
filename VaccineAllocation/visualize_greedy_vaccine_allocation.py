import pickle
import os
import pandas as pd
from datetime import datetime as dt
import numpy as np
from VaccineAllocation import load_config_file,config_path
from matplotlib import pyplot as plt
import calendar as py_cal
#from reporting.plotting import plot_multi_tier_sims, stack_plot
breakpoint()

def read_hosp(file_path, start_date, typeInput="hospitalized"):
    with open(file_path, 'r') as hosp_file:
        df_hosp = pd.read_csv(
            file_path,
            parse_dates=['date'],
            date_parser=pd.to_datetime,
        )
    # if hospitalization data starts before start_date 
    if df_hosp['date'][0] <= start_date:
        df_hosp = df_hosp[df_hosp['date'] >= start_date]
        real_hosp = list(df_hosp[typeInput])
    else:
        real_hosp = [0] * (df_hosp['date'][0] - start_date).days + list(df_hosp[typeInput])
    
    return real_hosp

def vaccine_spaghetti(file_path, instance_name, real_hosp=None, real_admit=None, hosp_beds_list=None, icu_beds_list=None, real_icu=None, 
                      iht_limit=None, icu_limit=None, toiht_limit=None, toicu_limit=None, t_start = -1, to_email=None):
    
    repno = 19 # 12
    # Read data
    with open(file_path, 'rb') as outfile:
        read_output = pickle.load(outfile)
    #instance, interventions, vaccines, stoch_replicas, selected_vaccine_policy, expected_cost, no_simulation_run, config = read_output
    instance, interventions, vaccines, stoch_replicas, sim_output, selected_vaccine_policy, expected_cost, no_simulation_run, config, seeds_info = read_output
    #instance, interventions, best_params, best_policy, best_sim, stoch_replicas, config, cost_record, seeds_info = read_output
    # Get only desired profiles
    if real_hosp is None:
        real_hosp = instance.cal.real_hosp
   
    profiles = [p for p in stoch_replicas]

    T = np.minimum(instance.T, instance.T)  #229
    
    states_to_plot = ['IHT', 'ICU', 'ToIHT']
    states_ts = {v: np.vstack(list(np.sum(p[v], axis=(1, 2))[:T] for p in profiles)) for v in states_to_plot}
    
    compartment_names = {
        'IY': 'Symptomatic',
        'IH': 'General Beds',
        'ToIHT': 'Daily Admissions',
        'D': 'Deaths',
        'R': 'Recovered',
        'S': 'Susceptible',
        'ICU': 'ICU Hospitalizations',
        'IHT': 'Total Hospitalizations',
        'ToICU': 'Daily ICU admissions'
    }
    
    print('vac example:', vaccines.vaccine_file_name)

    states_to_check = ['IY', 'D']
    states_ts_check = {v: np.vstack(list(np.sum(p[v], axis=(1, 2))[:T] for p in profiles)) for v in states_to_check}
    
    print('IY:', np.mean(states_ts_check['IY']))
    print('D:', np.mean(states_ts_check['D']))
    for v in states_to_plot: 
        ubbound = np.percentile(states_ts[v], 97.5, axis=0)
        lbbound = np.percentile(states_ts[v], 2.5, axis=0)
        #mean_val = states_ts[v].mean(axis = 0)
        mean_val = np.percentile(states_ts[v], 50, axis=0) #states_ts[v][repno]
        #std_val = states_ts[v].std(axis = 0)
        #error = 1.96*std_val
        
        if v == 'ToIHT':
            x = np.linspace(0, T-1, T-1)
        else:
            x = np.linspace(0, T, T)
        
        label = compartment_names[v]
        if v == 'ToIHT':
            real_data = real_admit 
        elif v == 'IHT':
            real_data = real_hosp
        elif v == 'ICU':
            real_data = real_icu
            
        plt.rcParams["font.size"] = "18"
        fig, ax1 = plt.subplots(1, 1, figsize=(17, 9))
        text_size = 18
        cal = instance.cal
        ax1.scatter(range(len(real_data)), real_data, color='maroon', label='Actual hospitalizations',zorder=100, s=15)

        plt.plot(x, mean_val, 'k-')
        plt.fill_between(x, lbbound, ubbound, color='b', alpha=0.2)
        plt.fill_between(x, lbbound, ubbound, color='b', alpha=0.2)
        ax1.set_ylabel(label)

        ax1.xaxis.set_ticks([t for t, d in enumerate(cal.calendar) if (d.day == 1 and t < T)])
        ax1.xaxis.set_ticklabels(
            [f' {py_cal.month_abbr[d.month]} ' for t, d in enumerate(cal.calendar) if (d.day == 1 and t < T)],
            rotation=0,
            fontsize=22)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label1.set_horizontalalignment('left')
        ax1.tick_params(axis='y', labelsize=text_size, length=5, width=2)
        ax1.tick_params(axis='x', length=5, width=2)
        
        t_start = 318 # vaccine starting time
        #max_y_lim_1 = np.max(mean_val+error)
        max_y_lim_1 = np.max(ubbound)
        plt.vlines(t_start, 0, max_y_lim_1, colors='k', linewidth = 5, linestyle='dashed')
         
        plt.show()
        
def icu_pipeline(file_path, instance_name, real_hosp=None, real_admit=None, hosp_beds_list=None, icu_beds_list=None, real_icu=None, 
                 iht_limit=None, icu_limit=None, toiht_limit=None, toicu_limit=None, t_start = -1, to_email=None):
    
    repno = 19 # 12
    # Read data
    with open(file_path, 'rb') as outfile:
        read_output = pickle.load(outfile)
    #instance, interventions, vaccines, stoch_replicas, selected_vaccine_policy, expected_cost, no_simulation_run, config = read_output
    instance, interventions, vaccines, stoch_replicas, sim_output, selected_vaccine_policy, expected_cost, no_simulation_run, config, seeds_info = read_output
    #instance, interventions, best_params, best_policy, best_sim, stoch_replicas, config, cost_record, seeds_info = read_output
    # Get only desired profiles
    if real_hosp is None:
        real_hosp = instance.cal.real_hosp
   
    profiles = [p for p in stoch_replicas]

    T = np.minimum(instance.T, instance.T)  #229
    
    states_to_plot = ['IHT', 'ICU', 'ToIHT']
    states_ts = {v: np.vstack(list(np.sum(p[v], axis=(1, 2))[:T] for p in profiles)) for v in states_to_plot}
    
    compartment_names = {
        'IY': 'Symptomatic',
        'IH': 'General Beds',
        'ToIHT': 'Daily Admissions',
        'D': 'Deaths',
        'R': 'Recovered',
        'S': 'Susceptible',
        'ICU': 'ICU Hospitalizations',
        'IHT': 'Total Hospitalizations',
        'ToICU': 'Daily ICU admissions'
    }
    
    print('vac example:', vaccines.vaccine_file_name)

    states_to_check = ['IY', 'D']
    states_ts_check = {v: np.vstack(list(np.sum(p[v], axis=(1, 2))[:T] for p in profiles)) for v in states_to_check}
    
    print('IY:', np.mean(states_ts_check['IY']))
    print('D:', np.mean(states_ts_check['D']))
    for v in states_to_plot: 
        ubbound = np.percentile(states_ts[v], 97.5, axis=0)
        lbbound = np.percentile(states_ts[v], 2.5, axis=0)
        mean_val = np.percentile(states_ts[v], 50, axis=0) #states_ts[v][repno]
        
        if v == 'ToIHT':
            x = np.linspace(0, T-1, T-1)
        else:
            x = np.linspace(0, T, T)
        
        label = compartment_names[v]
        if v == 'ToIHT':
            real_data = real_admit 
        elif v == 'IHT':
            real_data = real_hosp
        elif v == 'ICU':
            real_data = real_icu
            
        plt.rcParams["font.size"] = "18"
        fig, (ax1, actions_ax) = plt.subplots(2, 1, figsize=(17, 9), gridspec_kw={'height_ratios': [10, 1.1]})
        #fig, ax1 = plt.subplots(1, 1, figsize=(17, 9))
        text_size = 18
        cal = instance.cal
        ax1.scatter(range(len(real_data)), real_data, color='maroon', label='Actual hospitalizations',zorder=100, s=15)

        ax1.plot(x, mean_val, 'k-')
        ax1.fill_between(x, lbbound, ubbound, color='b', alpha=0.2)
        ax1.fill_between(x, lbbound, ubbound, color='b', alpha=0.2)
        ax1.set_ylabel(label)

        ax1.xaxis.set_ticks([t for t, d in enumerate(cal.calendar) if (d.day == 1 and t < T)])
        ax1.xaxis.set_ticklabels(
            [f' {py_cal.month_abbr[d.month]} ' for t, d in enumerate(cal.calendar) if (d.day == 1 and t < T)],
            rotation=0,
            fontsize=22)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label1.set_horizontalalignment('left')
        ax1.tick_params(axis='y', labelsize=text_size, length=5, width=2)
        ax1.tick_params(axis='x', length=5, width=2)
   
        actions_ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            left=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False)  # labels along the bottom edge are off
        actions_ax.spines['top'].set_visible(False)
        actions_ax.spines['bottom'].set_visible(False)
        actions_ax.spines['left'].set_visible(False)
        actions_ax.spines['right'].set_visible(False)
        
        if 140 <= T:
            actions_ax.annotate('2020',
                                xy=(140, 0),
                                xycoords='data',
                                color='k',
                                annotation_clip=True,
                                fontsize=text_size - 2)
        if 425 <= T:
            actions_ax.annotate('2021',
                                xy=(425, 0),
                                xycoords='data',
                                color='k',
                                annotation_clip=True,
                                fontsize=text_size - 2)
        
        t_start = 318 # vaccine starting time
        max_y_lim_1 = np.max(ubbound)
        plt.vlines(t_start, 0, max_y_lim_1, colors='k', linewidth = 5, linestyle='dashed')
         
        plt.show()
    
def det_plot(file_path, instance_name, real_hosp=None, real_admit=None, real_death=None, real_death_total=None, hosp_beds_list=None, icu_beds_list=None, real_icu=None, 
                 iht_limit=None, icu_limit=None, toiht_limit=None, toicu_limit=None, t_start = -1, to_email=None):
    
    # Read data
    with open(file_path, 'rb') as outfile:
        read_output = pickle.load(outfile)
    #instance, interventions, vaccines, stoch_replicas, selected_vaccine_policy, expected_cost, no_simulation_run, config = read_output
    instance, interventions, vaccines, stoch_replicas, sim_output, selected_vaccine_policy, expected_cost, no_simulation_run, config, seeds_info = read_output
    #instance, interventions, best_params, best_policy, best_sim, stoch_replicas, config, cost_record, seeds_info = read_output
    # Get only desired profiles
    if real_hosp is None:
        real_hosp = instance.cal.real_hosp
   
    T = np.minimum(instance.T, instance.T)  #229
    
    states_to_plot = ['IHT', 'ICU', 'ToIHT', 'ToICUD', 'D', 'ToIYD']
    states_ts = {v: np.vstack(list(np.sum(sim_output[v], axis=(1, 2))[:T])) for v in states_to_plot}
    
    compartment_names = {
        'IY': 'Symptomatic',
        'IH': 'General Beds',
        'ToIHT': 'Daily Admissions',
        'ToICUD': 'Daily Hospital Deaths',
        'ToIYD': 'Daily Non-Hospital Deaths',
        'R': 'Recovered',
        'S': 'Susceptible',
        'D': 'Cumulative Deaths',
        'ICU': 'ICU Hospitalizations',
        'IHT': 'Total Hospitalizations',
        'ToICU': 'Daily ICU admissions'
    }
    
    print('vac example:', vaccines.vaccine_file_name)
    
    for v in states_to_plot: 
        
        #mean_val = states_ts[v].mean(axis = 0)
        #std_val = states_ts[v].std(axis = 0)
        #error = 1.96*std_val
        

        x = np.linspace(0, T, T)
        
        label = compartment_names[v]
        if v == 'ToIHT':
            real_data = real_admit 
            x = np.linspace(0, T-1, T-1)
        elif v == 'IHT':
            real_data = real_hosp
        elif v == 'ICU':
            real_data = real_icu
        elif v == 'ToICUD':
            real_data = real_death
            x = np.linspace(0, T-1, T-1)
        elif v == 'ToIYD':
            #real_data = real_death
            
            real_data = [a_i - b_i for a_i, b_i in zip(real_death_total, real_death)]
            del real_data[0]
            x = np.linspace(0, T-1, T-1)            
            #daily_toIYD_benchmark = [sim_output['ToIYD'][t].sum() for t in range(len(instance.cal.real_hosp) - 1)] 
            #daily_death_benchmark.insert(0, 0)
            #x = np.linspace(0, T-1, T-1)            
        elif v == 'D':
            real_data = real_death_total
   
            

        if v == 'D':
            plt.rcParams["font.size"] = "18"
            fig, ax1 = plt.subplots(1, 1, figsize=(17, 9))
            text_size = 18
            cal = instance.cal
            ax1.scatter(range(len(real_data)), np.cumsum(real_data), color='maroon', label='Actual hospitalizations',zorder=100, s=15)


            plt.plot(x, states_ts[v], 'k-')
            #plt.fill_between(x, np.maximum(0, mean_val-error), mean_val+error, color='b', alpha=0.2)
            ax1.set_ylabel(label)
    
            ax1.xaxis.set_ticks([t for t, d in enumerate(cal.calendar) if (d.day == 1 and t < T)])
            ax1.xaxis.set_ticklabels(
                [f' {py_cal.month_abbr[d.month]} ' for t, d in enumerate(cal.calendar) if (d.day == 1 and t < T)],
                rotation=0,
                fontsize=22)
            for tick in ax1.xaxis.get_major_ticks():
                tick.label1.set_horizontalalignment('left')
            ax1.tick_params(axis='y', labelsize=text_size, length=5, width=2)
            ax1.tick_params(axis='x', length=5, width=2)
            
            #t_start = 370 # 308 # vaccine starting time
            #max_y_lim_1 = np.max(states_ts[v])
            #plt.vlines(t_start, 0, max_y_lim_1, colors='k', linewidth = 5, linestyle='dashed')
             
            plt.show()            
            
            #series = states_ts[v]
            #diff = np.zeros((len(series)))
            #for i in range(len(series)-1):
            #    diff[i] = series[i + 1] - series[i]
            #plt.plot(x, diff, 'k-')
            #ax1.set_ylabel(label)
    
            #ax1.xaxis.set_ticks([t for t, d in enumerate(cal.calendar) if (d.day == 1 and t < T)])
            #ax1.xaxis.set_ticklabels(
            #    [f' {py_cal.month_abbr[d.month]} ' for t, d in enumerate(cal.calendar) if (d.day == 1 and t < T)],
            #    rotation=0,
            #    fontsize=22)
            #for tick in ax1.xaxis.get_major_ticks():
           #     tick.label1.set_horizontalalignment('left')
           # ax1.tick_params(axis='y', labelsize=text_size, length=5, width=2)
          #  ax1.tick_params(axis='x', length=5, width=2)
            
           # t_start = 370 # 308 # vaccine starting time
            #max_y_lim_1 = np.max(diff)
            #plt.vlines(t_start, 0, max_y_lim_1, colors='k', linewidth = 5, linestyle='dashed')
             
            #plt.show()
        else:
            plt.rcParams["font.size"] = "18"
            fig, ax1 = plt.subplots(1, 1, figsize=(17, 9))
            text_size = 18
            cal = instance.cal
            ax1.scatter(range(len(real_data)), real_data, color='maroon', label='Actual hospitalizations',zorder=100, s=15)

            
            plt.plot(x, states_ts[v], 'k-')
            #plt.fill_between(x, np.maximum(0, mean_val-error), mean_val+error, color='b', alpha=0.2)
            ax1.set_ylabel(label)
    
            ax1.xaxis.set_ticks([t for t, d in enumerate(cal.calendar) if (d.day == 1 and t < T)])
            ax1.xaxis.set_ticklabels(
                [f' {py_cal.month_abbr[d.month]} ' for t, d in enumerate(cal.calendar) if (d.day == 1 and t < T)],
                rotation=0,
                fontsize=22)
            for tick in ax1.xaxis.get_major_ticks():
                tick.label1.set_horizontalalignment('left')
            ax1.tick_params(axis='y', labelsize=text_size, length=5, width=2)
            ax1.tick_params(axis='x', length=5, width=2)
            
            #t_start = 370 # 308 # vaccine starting time
            #max_y_lim_1 = np.max(states_ts[v])
            #plt.vlines(t_start, 0, max_y_lim_1, colors='k', linewidth = 5, linestyle='dashed')
             
            plt.show()
 

def death_rate(file_path, instance_name, real_hosp=None, real_admit=None, hosp_beds_list=None, icu_beds_list=None, real_icu=None, 
                 iht_limit=None, icu_limit=None, toiht_limit=None, toicu_limit=None, t_start = -1, to_email=None):
    
    repno = 19 # 12
    # Read data
    with open(file_path, 'rb') as outfile:
        read_output = pickle.load(outfile)
    #instance, interventions, vaccines, stoch_replicas, selected_vaccine_policy, expected_cost, no_simulation_run, config = read_output
    instance, interventions, vaccines, stoch_replicas, sim_output, selected_vaccine_policy, expected_cost, no_simulation_run, config, seeds_info = read_output
    #instance, interventions, best_params, best_policy, best_sim, stoch_replicas, config, cost_record, seeds_info = read_output
    # Get only desired profiles
    if real_hosp is None:
        real_hosp = instance.cal.real_hosp
   
    profiles = [p for p in stoch_replicas]

    T = np.minimum(instance.T, instance.T)  #229
    
    states_to_plot = ['D', 'ICU', 'ToICU', 'R']
    states_ts = {v: np.vstack(list(np.sum(p[v], axis=(1, 2))[:T] for p in profiles)) for v in states_to_plot}
    
    compartment_names = {
        'IY': 'Symptomatic',
        'IH': 'General Beds',
        'ToIHT': 'Daily Admissions',
        'D': 'Deaths',
        'R': 'Recovered',
        'S': 'Susceptible',
        'ICU': 'ICU Hospitalizations',
        'IHT': 'Total Hospitalizations',
        'ToICU': 'Daily ICU admissions'
    }
    
    print('vac example:', vaccines.vaccine_file_name)

    plt.rcParams["font.size"] = "18"
    D = states_ts['D'].mean(axis=0)
    ICU = states_ts['ICU'].mean(axis=0)
    ToICU = states_ts['ToICU'].mean(axis=0)
    ToD = np.diff(D)
    cumToD = np.cumsum(ToD)
    cumToICU = np.cumsum(ToICU) 
    fig, ax1 = plt.subplots(1, 1, figsize=(17, 9))
    ax1.scatter(range(len(D)), D, zorder=100, s=25)
    plt.xlabel("Time (days)")
    plt.ylabel("D compartment")        
    plt.show()
    fig, ax1 = plt.subplots(1, 1, figsize=(17, 9))
    ax1.scatter(range(len(ICU)), ICU, zorder=100, s=25)
    plt.xlabel("Time (days)")
    plt.ylabel("ICU compartment")        
    plt.show()
    fig, ax1 = plt.subplots(1, 1, figsize=(17, 9))
    ax1.scatter(range(len(cumToICU)), cumToICU, zorder=100, s=25)
    plt.xlabel("Time (days)")
    plt.ylabel("cumToICU")        
    plt.show()
    fig, ax1 = plt.subplots(1, 1, figsize=(17, 9))
    ax1.scatter(range(len(ToD)), ToD, zorder=100, s=25)
    plt.xlabel("Time (days)")
    plt.ylabel("Daily Death admission (ToD)")        
    plt.show()
    fig, ax1 = plt.subplots(1, 1, figsize=(17, 9))
    ax1.scatter(range(len(ToICU)), ToICU, zorder=100, s=25)
    plt.xlabel("Time (days)")
    plt.ylabel("Daily ICU admission (ToICU)")        
    plt.show()
    fig, ax1 = plt.subplots(1, 1, figsize=(17, 9))
    ax1.scatter(range(len(ToD)), (ToD/ToICU)*100, zorder=100, s=25)
    plt.xlabel("Time (days)")
    plt.ylabel("ToD/ToICU (percentage)")  
    plt.show()
    fig, ax1 = plt.subplots(1, 1, figsize=(17, 9))
    ax1.scatter(range(len(cumToD)), (cumToD/cumToICU)*100, zorder=100, s=25)
    plt.xlabel("Time (days)")
    plt.ylabel("cumsum(ToD)/cumsum(ToICU) (percentage)")  
    plt.show()
    fig, ax1 = plt.subplots(1, 1, figsize=(17, 9))
    ax1.scatter(range(len(D)), (D/(D + ICU))*100, zorder=100, s=25)
    plt.xlabel("Time (days)")
    plt.ylabel("D/(D+ICU) (percentage)")  
    plt.show()
        # ubbound = np.percentile(states_ts[v], 97.5, axis=0)
        # lbbound = np.percentile(states_ts[v], 2.5, axis=0)
        # #mean_val = states_ts[v].mean(axis = 0)
        # mean_val = np.percentile(states_ts[v], 50, axis=0) #states_ts[v][repno]
        # #std_val = states_ts[v].std(axis = 0)
        # #error = 1.96*std_val
        
        # if v == 'ToIHT':
        #     x = np.linspace(0, T-1, T-1)
        # else:
        #     x = np.linspace(0, T, T)
        
        # label = compartment_names[v]
        # if v == 'ToIHT':
        #     real_data = real_admit 
        # elif v == 'IHT':
        #     real_data = real_hosp
        # elif v == 'ICU':
        #     real_data = real_icu
            
        # plt.rcParams["font.size"] = "18"
        # fig, ax1 = plt.subplots(1, 1, figsize=(17, 9))
        # text_size = 18
        # cal = instance.cal
        # ax1.scatter(range(len(real_data)), real_data, color='maroon', label='Actual hospitalizations',zorder=100, s=15)

        # plt.plot(x, mean_val, 'k-')
        # plt.fill_between(x, lbbound, ubbound, color='b', alpha=0.2)
        # plt.fill_between(x, lbbound, ubbound, color='b', alpha=0.2)
        # ax1.set_ylabel(label)

        # ax1.xaxis.set_ticks([t for t, d in enumerate(cal.calendar) if (d.day == 1 and t < T)])
        # ax1.xaxis.set_ticklabels(
        #     [f' {py_cal.month_abbr[d.month]} ' for t, d in enumerate(cal.calendar) if (d.day == 1 and t < T)],
        #     rotation=0,
        #     fontsize=22)
        # for tick in ax1.xaxis.get_major_ticks():
        #     tick.label1.set_horizontalalignment('left')
        # ax1.tick_params(axis='y', labelsize=text_size, length=5, width=2)
        # ax1.tick_params(axis='x', length=5, width=2)
        
        # t_start = 318 # vaccine starting time
        # #max_y_lim_1 = np.max(mean_val+error)
        # max_y_lim_1 = np.max(ubbound)
        # plt.vlines(t_start, 0, max_y_lim_1, colors='k', linewidth = 5, linestyle='dashed')
         
        # plt.show()
        
if __name__ == "__main__":
    # list all .p files from the output folder
    #folder_name = "Benchmarks"
    #fileList = os.listdir("output/{}".format(folder_name))
    fileList = os.listdir("output")
    for instance_raw in fileList:
        if ".p" in instance_raw:
            if "austin" in instance_raw:
                file_path = "instances/austin/austin_real_hosp_updated.csv"
                start_date = dt(2020,2,28)
                real_hosp = read_hosp(file_path, start_date)
                hosp_beds_list = None
                file_path = "instances/austin/austin_hosp_ad_updated.csv"
                hosp_ad = read_hosp(file_path, start_date, "admits")
                file_path = "instances/austin/austin_real_icu_updated.csv"
                real_icu = read_hosp(file_path, start_date)
                file_path = "instances/austin/austin_real_death_from_hosp_updated.csv"
                real_death = read_hosp(file_path, start_date)   
                file_path = "instances/austin/austin_real_total_death.csv"
                real_death_total = read_hosp(file_path, start_date)                       
                iht_limit = 2000
                icu_limit = 500
                toiht_limit = 250
                toicu_limit = 100
                hosp_beds_list = [1500]
                icu_beds_list = [331]
                t_start = 193
            elif "houston" in instance_raw:
                file_path = "instances/houston/houston_real_hosp_updated.csv"
                start_date = dt(2020,2,19)
                real_hosp = read_hosp(file_path, start_date)
                hosp_beds_list = None
                file_path = "instances/houston/houston_real_icu_updated.csv"
                real_icu = read_hosp(file_path, start_date)
                hosp_ad = None
                if "tiers1" in instance_raw:
                    hosp_beds_list = [4500,9000,13500]
                else:
                    hosp_beds_list = None
                hosp_ad = None
                iht_limit = 6000
                icu_limit = 2000
                toiht_limit = 700
                toicu_limit = 200
                hosp_beds_list = [4500]
                icu_beds_list = [1250]
                t_start = 202
                
            instance_name = instance_raw[:-2]
            #path_file = f'output/{folder_name}/{instance_name}.p'
            path_file = f'output/{instance_name}.p'
            #multi_tier_pipeline(path_file, instance_name, real_hosp, hosp_ad, hosp_beds_list)
            
            # death_rate(path_file, instance_name, real_hosp, hosp_ad, hosp_beds_list, icu_beds_list, real_icu,
            #             iht_limit, icu_limit, toiht_limit, toicu_limit, t_start)
            
            icu_pipeline(path_file, instance_name, real_hosp, hosp_ad, hosp_beds_list, icu_beds_list, real_icu,
                           iht_limit, icu_limit, toiht_limit, toicu_limit, t_start)
            #det_plot(path_file, instance_name, real_hosp, hosp_ad, real_death, real_death_total, hosp_beds_list, icu_beds_list, real_icu,
            #         iht_limit, icu_limit, toiht_limit, toicu_limit, t_start)            
# fitted_ratio = ICUList/(ICUList + IHList)
# plt.plot(fitted_ratio[:150])
# plt.scatter(list(range(len(real_ratio))),real_ratio,color="crimson")