import pandas as pd
df = pd.read_csv("data\\processed\\churn_clean.csv")

from sklearn.model_selection import train_test_split
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
numeric_cols = ['tenure','MonthlyCharges','TotalCharges']
X_train[numeric_cols]=scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols]=scaler.transform(X_test[numeric_cols])

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train) 
y_pred=model.predict(X_test)

import joblib
joblib.dump(model, "model/model_logistic_regression.pkl")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }
results = evaluate_model(y_test, y_pred)
print("\n\n\tLogistic Regression Model Evaluation Results:")
print(f"Accuracy: {results['accuracy']:.2f}")   
print(f"Precision: {results['precision']:.2f}")   
print(f"Recall: {results['recall']:.2f}")   
print(f"F1 Score: {results['f1_score']:.2f}")  
print("Confusion Matrix: ",results['confusion_matrix'])

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
ConfusionMatrixDisplay(results['confusion_matrix']).plot()
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig("report/logistic_confusion_matrix.png")
plt.show()

from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(random_state=42)  
rf_model.fit(X_train,y_train)  
rf_pred=rf_model.predict(X_test)

joblib.dump(rf_model, "model/model_random_forest.pkl")

rf_results = evaluate_model(y_test, rf_pred)

print("\n\n\tRandom Forest Model Evaluation Results:")
print(f"Accuracy: {rf_results['accuracy']:.2f}")   
print(f"Precision: {rf_results['precision']:.2f}")   
print(f"Recall: {rf_results['recall']:.2f}")   
print(f"F1 Score: {rf_results['f1_score']:.2f}")  
print("Confusion Matrix: ",rf_results['confusion_matrix'])
ConfusionMatrixDisplay(rf_results['confusion_matrix']).plot()

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("report/random_forest_confusion_matrix.png")
plt.show()

importance = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importance
})
feature_importance = feature_importance.sort_values(by="importance", ascending=False)
print("\n\n\tTop 10 Important Features:")
print(feature_importance.head(10))
feature_importance.head(10).plot(kind='barh', x='feature', y='importance')
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig("report/random_forest_feature_importance.png")
plt.show()

from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train,y_train)  
dt_pred=dt_model.predict(X_test)

joblib.dump(dt_model, "model/model_decision_tree.pkl")