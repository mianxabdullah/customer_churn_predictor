import pandas as pd
df=pd.read_csv("data\\raw\\churn.csv")
print(df.isnull().sum())
print(df.describe())
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
print(df['TechSupport'].unique())
df=df.drop(columns=['customerID'])

#for eda
df_raw=df.copy()

