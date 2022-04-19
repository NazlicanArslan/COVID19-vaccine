import pickle
import os
import pandas as pd
from datetime import datetime as dt
import numpy as np
from VaccineAllocation import load_config_file,config_path
from reporting.plotting_CDC import plot_multi_tier_sims, stack_plot
from reporting.report_pdf import generate_report
from reporting.output_processors import build_report,build_report_tiers


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
    
def icu_pipeline(file_path, instance_name, real_hosp=None, real_admit=None, hosp_beds_list=None, icu_beds_list=None, real_icu=None, 
                 iht_limit=None, icu_limit=None, toiht_limit=None, toicu_limit=None, t_start = -1, to_email=None, is_representative_path_bool=False,
                 central_id_path = 0, cap_id_path = 0, acs_type = 'IHT'):
    
    # Read data
    print(file_path)
    with open(file_path, 'rb') as outfile:
        read_output = pickle.load(outfile)
    instance, interventions, best_params, best_policy, vaccines, profiles, sim_output, expected_cost, config, seeds_info = read_output

    # Get only desired profiles
    if real_hosp is None:
        real_hosp = instance.cal.real_hosp
    last_day_hosp_data = len(real_hosp) - 1
    lb_hosp = real_hosp[-1] * (1 - config['div_filter_frac'])
    ub_hosp = real_hosp[-1] * (1 + config['div_filter_frac'])
   
    profiles = [p for p in profiles]

    n_replicas = len(profiles)
    T = np.minimum(instance.T, instance.T)  
    
    plot_trigger_ToICU = False
    plot_trigger_ToIHT = False
    plot_trigger_ICU = False
    plot_trigger_ToIHT = True
     
    moving_avg_len = config['moving_avg_len']
    N = instance.N
    icu_beds_list = [instance.icu]
    real_icu_ratio = [real_icu[i]/(real_hosp[i]) for i in range(len(real_icu))  if real_hosp[i] != 0]
    real_ToIHT_total = [np.array(real_admit)[i: min(i + moving_avg_len, T)].sum()* 100000/np.sum(N, axis=(0,1))  for i in range(T-moving_avg_len)]
    real_percent_IH = [np.array(real_hosp)[i: min(i + moving_avg_len, T)].mean()/instance.hosp_beds for i in range(T-moving_avg_len)]
    real_case_total = [np.array(case_ad)[i: min(i + moving_avg_len, T)].sum()* 100000/np.sum(N, axis=(0,1)) for i in range(T-moving_avg_len)]
    # plot the IHT comparison
    # IHD_plot = plot_multi_tier_sims(instance_name,
    #                         instance,
    #                         best_policy,
    #                         profiles, ['sim'] * len(profiles),
    #                         real_icu,
    #                         plot_left_axis=['ICU'],
    #                         plot_right_axis=[],
    #                         T=T,
    #                         interventions=interventions,
    #                         show=True,
    #                         align_axes=True,
    #                         plot_triggers=plot_trigger_ICU,
    #                         plot_trigger_annotations=False,
    #                         plot_legend=False,
    #                         y_lim=icu_limit,
    #                         policy_params=best_params,
    #                         n_replicas=n_replicas,
    #                         config=config,
    #                         hosp_beds_list= icu_beds_list,
    #                         real_new_admission=real_admit,
    #                         real_new_admission_unvax = None,
    #                         real_hosp_or_icu=real_icu,
    #                         t_start = t_start,
    #                         is_representative_path=is_representative_path_bool,
    #                         central_path_id = central_id_path,
    #                         cap_path_id = cap_id_path,
    #                         vertical_fill = not plot_trigger_ICU,
    #                         history_white = True,
    #                         acs_type = acs_type
    #                         )

    IYIH_plot2 = plot_multi_tier_sims(instance_name,
                            instance,
                            best_policy,
                            profiles, ['sim'] * len(profiles),
                            real_hosp,
                            plot_left_axis=['ToIY_moving'],
                            plot_right_axis=[],
                            T=T,
                            interventions=interventions,
                            show=True,
                            align_axes=True,
                            plot_triggers=plot_trigger_ToIHT,
                            plot_trigger_annotations=False,
                            plot_legend=False,
                            y_lim=1100,
                            policy_params=best_params,
                            n_replicas=n_replicas,
                            config=config,
                            hosp_beds_list=[200],
                            real_new_admission=real_case_total,
                            real_new_admission_unvax = None,
                            real_new_admission_vax =None, 
                            real_hosp_or_icu=real_case_total,
                            t_start = t_start,
                            is_representative_path=is_representative_path_bool,
                            central_path_id = central_id_path,
                            cap_path_id = cap_id_path,
                            vertical_fill = True,
                            nday_avg = 7,
                            history_white = True,
                            acs_type = acs_type
                            )   
    
    # plot the IHT comparison
    # IHD_plot2 = plot_multi_tier_sims(instance_name,
    #                         instance,
    #                         best_policy,
    #                         profiles, ['sim'] * len(profiles),
    #                         real_hosp,
    #                         plot_left_axis=['IHT'],
    #                         plot_right_axis=[],
    #                         T=T,
    #                         interventions=interventions,
    #                         show=True,
    #                         align_axes=True,
    #                         plot_triggers=False,
    #                         plot_trigger_annotations=False,
    #                         plot_legend=False,
    #                         y_lim=iht_limit,
    #                         policy_params=best_params,
    #                         n_replicas=n_replicas,
    #                         config=config,
    #                         hosp_beds_list= hosp_beds_list,
    #                         real_new_admission=real_admit,
    #                         real_new_admission_unvax = None,
    #                         real_new_admission_vax =None, 
    #                         real_hosp_or_icu=real_hosp,
    #                         t_start = t_start,
    #                         is_representative_path=is_representative_path_bool,
    #                         central_path_id = central_id_path,
    #                         cap_path_id = cap_id_path,
    #                         history_white = True,
    #                         acs_type = acs_type
    #                         )

    
    IYIH_plot2 = plot_multi_tier_sims(instance_name,
                            instance,
                            best_policy,
                            profiles, ['sim'] * len(profiles),
                            real_hosp,
                            plot_left_axis=['ToIHT_total'],
                            plot_right_axis=[],
                            T=T,
                            interventions=interventions,
                            show=True,
                            align_axes=True,
                            plot_triggers=plot_trigger_ToIHT,
                            plot_trigger_annotations=False,
                            plot_legend=False,
                            y_lim=60,
                            policy_params=best_params,
                            n_replicas=n_replicas,
                            config=config,
                            hosp_beds_list=None,
                            real_new_admission=real_ToIHT_total,
                            real_new_admission_unvax = None,
                            real_new_admission_vax =None, 
                            real_hosp_or_icu=real_ToIHT_total,
                            t_start = t_start,
                            is_representative_path=is_representative_path_bool,
                            central_path_id = central_id_path,
                            cap_path_id = cap_id_path,
                            vertical_fill = not plot_trigger_ToIHT,
                            nday_avg = 7,
                            history_white = True,
                            acs_type = acs_type
                            )   
    IYIH_plot2 = plot_multi_tier_sims(instance_name,
                            instance,
                            best_policy,
                            profiles, ['sim'] * len(profiles),
                            real_hosp,
                            plot_left_axis=['IHT_moving'],
                            plot_right_axis=[],
                            T=T,
                            interventions=interventions,
                            show=True,
                            align_axes=True,
                            plot_triggers=plot_trigger_ToIHT,
                            plot_trigger_annotations=False,
                            plot_legend=False,
                            y_lim=1,
                            policy_params=best_params,
                            n_replicas=n_replicas,
                            config=config,
                            hosp_beds_list=None,
                            real_new_admission=real_percent_IH,
                            real_new_admission_unvax = None,
                            real_new_admission_vax =None, 
                            real_hosp_or_icu=real_percent_IH,
                            t_start = t_start,
                            is_representative_path=is_representative_path_bool,
                            central_path_id = central_id_path,
                            cap_path_id = cap_id_path,
                            vertical_fill = not plot_trigger_ToIHT,
                            nday_avg = 7,
                            history_white = True,
                            acs_type = acs_type
                            )   
    
    IYIH_plot3 = plot_multi_tier_sims(instance_name,
                            instance,
                            best_policy,
                            profiles, ['sim'] * len(profiles),
                            real_hosp,
                            plot_left_axis=['ToIHT_moving'],
                            plot_right_axis=[],
                            T=T,
                            interventions=interventions,
                            show=True,
                            align_axes=True,
                            plot_triggers=real_admit,
                            plot_trigger_annotations=False,
                            plot_legend=False,
                            y_lim=toiht_limit,
                            policy_params=best_params,
                            n_replicas=n_replicas,
                            config=config,
                            hosp_beds_list=None,
                            real_new_admission=real_admit,
                            real_new_admission_unvax = None,
                            real_new_admission_vax =None, 
                            real_hosp_or_icu=real_ToIHT_total,
                            t_start = t_start,
                            is_representative_path=is_representative_path_bool,
                            central_path_id = central_id_path,
                            cap_path_id = cap_id_path,
                            vertical_fill = not plot_trigger_ToIHT,
                            nday_avg = 7,
                            history_white = True,
                            acs_type = acs_type
                            )   
    
 
    
    IHT_stacked_plot = stack_plot(instance_name+"_stacked",
                                  instance,
                                  best_policy,
                                  profiles, ['sim'] * len(profiles),
                                  real_hosp,
                                  plot_left_axis=['ICU'],
                                  plot_right_axis=[],
                                  T=T,
                                  interventions=interventions,
                                  show=False,
                                  align_axes=False,
                                  plot_triggers=False, #set false when looking at total hospitalizations
                                  plot_trigger_annotations=False,
                                  plot_legend=False,
                                  y_lim=icu_limit,
                                  policy_params=best_params,
                                  n_replicas=n_replicas,
                                  config=config,
                                  hosp_beds_list=icu_beds_list,
                                  real_new_admission=real_admit,
                                  real_icu_patients=real_icu,
                                  real_hosp_or_icu=real_icu,
                                  period=1,
                                  is_representative_path=False,
                                  t_start = t_start,
                                  central_path_id = central_id_path,
                                  cap_path_id = central_id_path,
                                  history_white = True)
    
    if hosp_beds_list is None:
        hosp_beds_list = [instance.hosp_beds]
    if icu_beds_list is None:
        icu_beds_list = [instance.icu]

if __name__ == "__main__":
    # list all .p files from the output folder
    fileList = os.listdir("output")
    for instance_raw in fileList:
        if ".p" in instance_raw:
            if "austin" in instance_raw:
                file_path = "instances/austin/austin_real_hosp_updated.csv"
                start_date = dt(2020,2,28)
                end_history = dt(2022,4,4)
                real_hosp = read_hosp(file_path, start_date)
                hosp_beds_list = None
                file_path = "instances/austin/austin_hosp_ad_updated.csv"
                hosp_ad = read_hosp(file_path, start_date, "admits")
                file_path = "instances/austin/austin_real_case.csv"
                case_ad = read_hosp(file_path, start_date, "admits")
                file_path = "instances/austin/austin_real_icu_updated.csv"
                real_icu = read_hosp(file_path, start_date)
                iht_limit = 2000
                icu_limit = 500
                toiht_limit = 150
                toicu_limit = 100
                hosp_beds_list = [1100]
                icu_beds_list = None
                t_start =(end_history - start_date).days
                central_id_path = 0
                acs_type = 'ICU'
                
            instance_name = instance_raw[:-2]
            path_file = f'output/{instance_name}.p'
            icu_pipeline(path_file, instance_name, real_hosp, hosp_ad, hosp_beds_list, icu_beds_list, real_icu,
                          iht_limit, icu_limit, toiht_limit, toicu_limit, t_start, None, False, central_id_path, central_id_path, acs_type)
