import pandas as pd
from pathlib import Path

url="https://www.dshs.state.tx.us/coronavirus/TexasCOVID19DailyCountyFatalityCountData.xlsx"

#get data frame
df_2020 = pd.read_excel(url,sheet_name=0, index_col=0,parse_dates=[0])
df_2020 = df_2020[1:]
df_2020=df_2020.T
df_2020=df_2020[["County", "Hays", "Bastrop", "Caldwell", "Travis", "Williamson"]]

df_2021 = pd.read_excel(url,sheet_name=1, index_col=0,parse_dates=[0])
df_2021 = df_2021[1:]
df_2021=df_2021.T
df_2021=df_2021[["County", "Hays", "Bastrop", "Caldwell", "Travis", "Williamson"]]


df_2022 = pd.read_excel(url,sheet_name=2, index_col=0,parse_dates=[0])
df_2022 = df_2022[1:]
df_2022=df_2022.T
df_2022=df_2022[["County", "Hays", "Bastrop", "Caldwell", "Travis", "Williamson"]]


df_all=df_2020.append(df_2021)
df_all=df_all.append(df_2022)

dict = {'County' : 'Date'}
df_all.rename(columns=dict,
          inplace=True)

instances_path = Path(__file__).parent
path = instances_path.parent / 'instances/austin/scraped_death_data.csv'

df_all.to_csv(path, index=False)

