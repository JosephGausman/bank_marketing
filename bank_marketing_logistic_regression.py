
#---Bank Marketing - Classification - Logistic Regression----------------------

#---Working Directory----------------------------------------------------------

import os
os.getcwd()
os.chdir(...)

#---Libraries------------------------------------------------------------------

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

#---Data loading---------------------------------------------------------------

raw_data = pd.read_csv("bank-additional-full.csv", sep = ';', na_values='NA')

raw_data.head()
raw_data.columns
raw_data.isnull()
raw_data.info()

# Remove observations where call duration equals zero--------------------------

clean_data=raw_data[(raw_data['duration'] != 0)]

#---Target Variable------------------------------------------------------------

y = pd.get_dummies(clean_data['y'], columns = ['y'], prefix = ['y'], drop_first = True)

#---Categorical Treatment------------------------------------------------------

client = clean_data.iloc[: , 0:7]

client.columns

labelencoder_X = LabelEncoder()
client['job']      = labelencoder_X.fit_transform(client['job']) 
client['marital']  = labelencoder_X.fit_transform(client['marital']) 
client['education']= labelencoder_X.fit_transform(client['education']) 
client['default']  = labelencoder_X.fit_transform(client['default']) 
client['housing']  = labelencoder_X.fit_transform(client['housing']) 
client['loan']     = labelencoder_X.fit_transform(client['loan']) 

client.head()

#---Grouping by age------------------------------------------------------------

def age(dataframe):
    dataframe.loc[dataframe['age'] <= 30, 'age'] = 1
    dataframe.loc[(dataframe['age'] > 30) & (dataframe['age'] <= 50), 'age'] = 2
    dataframe.loc[(dataframe['age'] > 50) & (dataframe['age'] <= 70), 'age'] = 3
    dataframe.loc[(dataframe['age'] > 70) & (dataframe['age'] <= 100), 'age'] = 4
           
    return dataframe

age(client);

client.head()

#---Contact, Month, Day of Week treatment--------------------------------------

contact = clean_data.iloc[: , 7:11]
contact.head()

contact['contact']     = labelencoder_X.fit_transform(contact['contact']) 
contact['month']       = labelencoder_X.fit_transform(contact['month']) 
contact['day_of_week'] = labelencoder_X.fit_transform(contact['day_of_week']) 

contact.head()

def duration(data):
    data.loc[data['duration'] <= 102, 'duration'] = 1
    data.loc[(data['duration'] > 102) & (data['duration'] <= 180)  , 'duration']    = 2
    data.loc[(data['duration'] > 180) & (data['duration'] <= 319)  , 'duration']   = 3
    data.loc[(data['duration'] > 319) & (data['duration'] <= 644.5), 'duration'] = 4
    data.loc[data['duration']  > 644.5, 'duration'] = 5

    return data

duration(contact);

contact.head()


#---Social and economic context attributes-------------------------------------

soc_eco = clean_data.loc[: , ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]
soc_eco.head()

#---Other attributes-----------------------------------------------------------

other = clean_data.loc[: , ['campaign', 'pdays','previous', 'poutcome']]
other.head()

other['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)
other.head()

#---Final Dataset--------------------------------------------------------------

bank_final= pd.concat([client, contact, soc_eco, other], axis = 1)
bank_final.head()

#---Splitting------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(bank_final, y, test_size = 0.2, random_state = 0)

#---Scailing-------------------------------------------------------------------

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#---Logistic Regression--------------------------------------------------------

model = LogisticRegression() 
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(round(accuracy_score(y_test, y_pred),2)*100)

#---ROC and AUC----------------------------------------------------------------

from sklearn.metrics import roc_curve, auc

y_pred_prob = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#---End------------------------------------------------------------------------
