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

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

#label encoding for binary categorical columns
df['Churn']=le.fit_transform(df['Churn'])
df['gender']=le.fit_transform(df['gender'])
# df['SeniorCitizen']=le.fit_transform(df['SeniorCitizen'])
df['Partner']=le.fit_transform(df['Partner'])
df['Dependents']=le.fit_transform(df['Dependents'])
df['PhoneService']=le.fit_transform(df['PhoneService'])
df['PaperlessBilling']=le.fit_transform(df['PaperlessBilling'])