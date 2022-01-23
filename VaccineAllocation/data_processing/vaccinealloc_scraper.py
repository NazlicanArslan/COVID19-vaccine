import pandas as pd
from pathlib import Path
from datetime import date
from datetime import timedelta

#Get most recent report date (assuming it was yesterday)
today = date.today()
yesterday = today - timedelta(days=1)
s=yesterday.strftime(("%Y-%m-%d"))
#url for excel file 
url="https://github.com/spencerwoody/texvax/blob/main/data/"+s+"%20COVID-19%20Vaccine%20Data%20by%20County.xlsx?raw=true"
#url1 = 'https://github.com/spencerwoody/texvax/blob/main/data/2022-01-13%20COVID-19%20Vaccine%20Data%20by%20County.xlsx?raw=true'

#get data frame
df = pd.read_excel(url,sheet_name=4, index_col=0,parse_dates=[0])


#convert to csv
instances_path = Path(__file__).parent
path = instances_path.parent / 'instances/austin/scraped_vaccine_data.csv'

df.to_csv(path)



