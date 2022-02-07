city = 'austin'
seeds_file_name = 'seed.p'
from pathlib import Path

instances_path = Path(__file__).parent

import pickle


# use this when we generate new seeds and want to write just seeds array
def write_seeds(city, seeds_file_name='seeds.p'):
    # read in the seeds file
    seedsinput = instances_path
    #breakpoint()
    print(seedsinput / seeds_file_name)
    try:
        with open(seedsinput / seeds_file_name, 'rb') as infile:
            seeds_data = pickle.load(infile)
            file_path = seedsinput / 'new_seed.p'
            with open(str(file_path), 'wb') as outfile:
                pickle.dump(((seeds_data[-1][0], seeds_data[-1][1]  )), 
                            outfile, pickle.HIGHEST_PROTOCOL)
        
        print(seeds_data[-1][0], seeds_data[-1][1]  )
        return seeds_data[-1][0], seeds_data[-1][1]    
        #return seeds_data['training'], seeds_data['testing']
    except:
        return [],[]

seeds_input_file_name='new_seed.p'
def load_seeds(city, seeds_file_name='newseed.p'):
    # read in the seeds file
    seedsinput = instances_path
    print(seedsinput / seeds_file_name)
    try:
        with open(seedsinput / seeds_file_name, 'rb') as infile:
            seeds_data = pickle.load(infile)

        return seeds_data[0], seeds_data[1]    

    except:
        return [],[]
    
    
write_seeds(city, seeds_file_name)
#print(load_seeds(city, seeds_input_file_name))