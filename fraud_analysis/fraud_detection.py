import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve,average_precision_score
import matplotlib.pyplot as plt
#load the dataset
df=pd.read_csv(r'creditcard.csv')
#take a look at the data (the first 5 samples)
print(df.head(5))

#check the class imbalance
print(df['Class'].value_counts())
print(df['Class'].value_counts(normalize=True)*100)

#get features X, and label y
X=df.drop('Class',axis=1)
y=df['Class']

#split train and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)

# Handle class imbalance: calculate class weights
#Weight_i = N / (Number of Classes * N_i) 
#a=df['Class'].value_counts()
#weight0,weight1=sum(a)/(2*a[0]),sum(a)/(2*a[1])
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

#creast and train a Random Forest model with class weights
model=RandomForestClassifier(n_estimators=20,random_state=0,class_weight=class_weight_dict,n_jobs=-1)
model.fit(X_train,y_train)

#Make prediction on the test set
y_pred=model.predict(X_test)

#Evaluate the model
print(classification_report(y_test, y_pred))

#Understand the results
'''The classification_report will show high accuracy but, more importantly, it will show Precision and Recall for the fraud class (Class 1).
              precision    recall  f1-score   support
           0       1.00      1.00      1.00     56864
           1       0.96      0.71      0.82        98
Precision = 0.96: Of the transactions we predicted as fraud, 95% were actually fraud.
Recall = 0.71: We successfully caught 80% of all the fraudulent transactions in the test set
'''
#average precision (AP) is a better evaluation metric than AUPR for imbalanced labels
print(average_precision_score(y_test, y_pred))
#plot precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.plot(recall,precision)
plt.xlabel('recall')
plt.ylabel('Precision')
plt.show()

