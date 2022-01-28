## running demo for COVID-19 multi-tier trigger analysis

## run the simulation for a fixed trigger policy:
python3 ../VaccineAllocation/main_allocation.py austin -f setup_data_Final.json -t tiers5_opt_Final.json -train_reps 0 -test_reps 1 -f_config austin_test_IHT.json -n_proc 1 -tr transmission_new.csv -hos austin_real_hosp_updated.csv  -v_allocation vaccine_allocation_fixed.csv -n_policy 7  -v_boost=booster_allocation_fixed.csv -gt [-1,5,15,30,50]

## perform the output procedure, plotting and generating pdf report
python3 ../VaccineAllocation/pipelinemultitier.py