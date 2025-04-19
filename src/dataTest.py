import pandas as pd

dataset_path = "resources/data.csv"

df = pd.read_csv(dataset_path)

print(df.loc[0,'text'])