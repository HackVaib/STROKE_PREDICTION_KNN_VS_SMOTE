import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE #as there is biasness in dataset towards non stroke patients

#data
df=pd.read_csv("healthcare-dataset-stroke-data.csv")

#preprocessing and handling missing values
df.drop('id',axis=1,inplace=True)

df['bmi'].fillna(df['bmi'].mean(),inplace=True)


le=LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col]=le.fit_transform(df[col])
    
X=df.drop('stroke',axis=1)
y=df['stroke']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)

#model without SMOTE
knn_plain=KNeighborsClassifier(n_neighbors=5)
knn_plain.fit(X_train,y_train)
y_pred_plain=knn_plain.predict(X_test)

#evaluation without smote
print("WITHOUT SMOTE:")
print("ACCURACY: ",accuracy_score(y_test,y_pred_plain))
print("CLASSIFICATION REPORT: ",classification_report(y_test,y_pred_plain))
print("Confusion Matrix : ", confusion_matrix(y_test, y_pred_plain))


#class imbalance handle with smote 
sm= SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

#model with smote
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_res,y_train_res)
y_pred=knn.predict(X_test)

#evaluation with smote
print("\nWITH SMOTE:")
print("ACCURACY: ",accuracy_score(y_test,y_pred))
print("CLASSIFICATION REPORT: ",classification_report(y_test,y_pred))
print("Confusion Matrix : ", confusion_matrix(y_test, y_pred))

#plotting
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.heatmap(confusion_matrix(y_test, y_pred_plain), annot=True, fmt='d', cmap='Reds')
plt.title("Without SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1,2,2)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.title("With SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()
