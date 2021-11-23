import pdb
pdb.set_trace()

if __name__ == '__main__':
    import multiprocessing as mp
    from utils import parse_arguments
    from VaccineAllocation import load_config_file, change_paths
    
    # Parse arguments
    args = parse_arguments()
    # Load config file
    load_config_file(args.f_config)
    # Adjust paths
    change_paths(args)
    
    from instances import load_instance, load_tiers, load_seeds, load_vaccines
    from objective_functions import multi_tier_objective
    from vaccine_policies import MultiTierPolicy as MTP
    from vaccine_policies import VaccineAllocationPolicy as VAP
    #from dfo_allocation_TW import run_allocation
    #from rolling_allocation import run_allocation
    from vaccine_policy_search_stochastic import stochastic_greedy_allocation
    
    # Parse city and get corresponding instance
    instance = load_instance(args.city, setup_file_name=args.f, transmission_file_name=args.tr, hospitalization_file_name=args.hos)
    train_seeds, test_seeds = load_seeds(args.city, args.seed)
    tiers = load_tiers(args.city, tier_file_name=args.t)
    #vaccines = load_vaccines(args.city, vaccine_file_name=args.v)
    vaccines = load_vaccines(args.city, vaccine_file_name=args.v, vaccine_allocation_file_name = args.v_allocation)
    
    # TODO Read command line args for n_proc for better integration with crunch
    n_proc = args.n_proc
    
    # TODO: pull out n_replicas_train and n_replicas_test to a config file
    n_replicas_train = args.train_reps
    n_replicas_test = args.test_reps
    
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
    
    given_allocation = eval(args.gv) if args.gv is not None else False
    given_threshold = eval(args.gt) if args.gt is not None else None
    given_date = eval('dt.datetime({})'.format(args.gd)) if args.gd is not None else None
    
    # if a threshold/threshold+stepping date is given, then it carries out a specific task
    # if not, then search for a policy
    selected_policy = None
    if tiers.tier_type == 'constant':
        if given_threshold is not None:
            selected_policy = MTP.constant_policy(instance, tiers.tier, given_threshold)
    elif tiers.tier_type == 'step':
        if (given_threshold is not None) and (given_date is not None):
            selected_policy = MTP.step_policy(instance, tiers.tier, given_threshold, given_date)
            
    # if a vaccine allocation is given, then it carries out a specific task
    # if not, then search for a policy
    selected_vaccine_policy = None
    if given_allocation is not False:
        selected_vaccine_policy = VAP.vaccine_policy(instance, vaccines)
    
    task_str = str(selected_policy) if selected_policy is not None else f'opt{len(tiers.tier)}'
    instance_name = f'{args.f_config[:-5]}_{args.t[:-5]}_{task_str}_rl{str(args.rl)}_{args.df_obj}_{args.is_qp}_{args.initial}'
    
    # read in the policy upper bound
    if args.pub is not None:
        policy_ub = eval(args.pub)
    else:
        policy_ub = None

    selected_vaccine_policy, obj_value = stochastic_greedy_allocation(instance=instance,
                                                           tiers=tiers.tier,
                                                           vaccines=vaccines,
                                                           obj_func=multi_tier_objective,
                                                           n_replicas_train=n_replicas_train,
                                                           n_replicas_test=n_replicas_test,
                                                           instance_name=instance_name,
                                                           policy_class=tiers.tier_type,
                                                           policy=selected_policy,
                                                           vaccine_policy=selected_vaccine_policy,
                                                           mp_pool=mp_pool,
                                                           crn_seeds=train_seeds,
                                                           unique_seeds_ori=test_seeds,
                                                           forcedOut_tiers=eval(args.fo),
                                                           redLimit=args.rl,
                                                           after_tiers=eval(args.aftert),
                                                           policy_field=args.field,
                                                           policy_ub=policy_ub,
                                                           dfo_obj=args.df_obj)
    
