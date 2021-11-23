# COVID19_CAOE Least-squares fit

To run the fit:

runfile('/main_least_squares.py', 'austin -f setup_data_Final_lsq.json -t tiers5_opt_Final.json -train_reps 1 -test_reps 1 -f_config austin_test_IHT.json -n_proc 1 -gt [-1,0,5,20,50] -tr transmission_Final_lsq.csv -hos austin_real_hosp_lsq.csv -v vaccine_supply_fixed_March4.csv -v_allocation vaccine_allocation_fixed_March4.csv')


The data sets are used in the fit should be adjusted based on the last of the fit (e.g., March, 04, 2021)


The end date of simulation should be picked from setup_data_Final_lsq.json based on that date.
