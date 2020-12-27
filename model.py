import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
import os
import pickle


#reading data
dataframe_raw = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv" )
testdata_raw =  pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_test.csv')

#preprocessing of data
df = dataframe_raw.copy(deep = True)
tdf = testdata_raw.copy(deep = True)
df = df.drop('Unnamed: 0', axis = 1)
df = df.drop('Loan_ID', axis=1)
df = pd.get_dummies(df, drop_first=True)

#splitting data and filling missing values
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify =y,random_state =42)
imp = SimpleImputer(strategy='mean')
imp_train = imp.fit(X_train)
X_train = imp_train.transform(X_train)
X_test = imp_train.transform(X_test)

#model 
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


print("Accuracy for Logistic Regression:", accuracy_score(y_test, y_pred))
print("F1 Score for Logistic Regression:", f1_score(y_test, y_pred))

print("Validation Mean F1 Score: ",cross_val_score(lr,X_train,y_train,cv=5,scoring='f1_macro').mean())
print("Validation Mean Accuracy: ",cross_val_score(lr,X_train,y_train,cv=5,scoring='accuracy').mean())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

train_accuracies = []
train_f1_scores = []
test_accuracies = []
test_f1_scores = []
thresholds = []

#for thresh in np.linspace(0.1,0.9,8): ## Sweeping from threshold of 0.1 to 0.9
for thresh in np.arange(0.1,0.9,0.1): ## Sweeping from threshold of 0.1 to 0.9
    logreg_clf = LogisticRegression(solver='liblinear', max_iter=1000)
    logreg_clf.fit(X_train,y_train)

    y_pred_train_thresh = logreg_clf.predict_proba(X_train)[:,1]
    y_pred_train = (y_pred_train_thresh > thresh).astype(int)

    train_acc = accuracy_score(y_train,y_pred_train)
    train_f1 = f1_score(y_train,y_pred_train)

    y_pred_test_thresh = logreg_clf.predict_proba(X_test)[:,1]
    y_pred_test = (y_pred_test_thresh > thresh).astype(int)

    test_acc = accuracy_score(y_test,y_pred_test)
    test_f1 = f1_score(y_test,y_pred_test)

    train_accuracies.append(train_acc)
    train_f1_scores.append(train_f1)
    test_accuracies.append(test_acc)
    test_f1_scores.append(test_f1)
    thresholds.append(thresh)


#Threshold_logreg = {"Training Accuracy": train_accuracies, "Test Accuracy": test_accuracies, "Training F1": train_f1_scores, "Test F1":test_f1_scores, "Decision Threshold": thresholds }
#Threshold_logreg_df = pd.DataFrame.from_dict(Threshold_logreg)

#plot_df = Threshold_logreg_df.melt('Decision Threshold',var_name='Metrics',value_name="Values")
#fig,ax = plt.subplots(figsize=(15,5))
#sns.pointplot(x="Decision Threshold", y="Values",hue="Metrics", data=plot_df,ax=ax)
thresh = 0.6 ### Threshold chosen from above Curves
y_pred_test_thresh = logreg_clf.predict_proba(X_test)[:,1]
y_pred = (y_pred_test_thresh > thresh).astype(int) 
print("Test Accuracy: ",accuracy_score(y_test,y_pred))
print("Test F1 Score: ",f1_score(y_test,y_pred))
#print("Confusion Matrix on Test Data")
#pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


#save the model 
filename = 'loan_prediction_v1.pkl'
pickle.dump(logreg_clf, open(filename, 'wb'))
