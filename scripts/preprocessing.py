"""
ML Project Workflow:
1. load and understand data (EDA)
2. preprocess data : handle missing values, data types
3. encode categorical features(columns) using label encoding or one hot encoding
4. train test split data into train and test set
5. scale data using standardscaler or minmaxscaler
6. train model using logistic regression or random forest classifier
7. evaluate model using accuracy, precision, recall, f1 score and confusion matrix
8. save model using joblib or pickle
9. visualize results using matplotlib or seaborn
"""
import pandas as pd
df=pd.read_csv("Project\\data\\raw\\churn.csv")
print("Null values in each column:")
print(df.isnull().sum())
print("\nData types and information:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe())
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
print("\nUnique values in 'TechSupport' column:")
print(df['TechSupport'].unique())
print("\nUnique values in 'InternetService' column:")
print(df['InternetService'].unique())
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

#one hot encoding for categorical columns with more than 2 categories
df=pd.get_dummies(df,columns=['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod'])

bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

print(df.dtypes)
print("Number of object columns:",len(df.select_dtypes(include=['object']).columns))

# save cleaned data for downstream scripts
df.to_csv("data\\processed\\churn_clean.csv", index=False)
print('Wrote churn_clean.csv with', len(df), 'rows')

df_raw.to_csv("data\\processed\\churn_eda.csv", index=False)
print('Wrote churn_eda.csv with', len(df_raw), 'rows')