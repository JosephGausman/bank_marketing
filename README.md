## Bank Marketing – Logistic Regression

The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution (May 2008 to November 2010).
The classification goal was to predict if the client will subscribe (yes/no) a term deposit (variable y).
The data set consists of 41118 observations and 21 attributes.
Source of the dataset: [http://archive.ics.uci.edu/ml/datasets/Bank+Marketing#](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

## Attributes

Clients data: Age, Job type, Marital status, Education, Default, Housing, Loan

Last contact: Contact, Month, Day of the week, Call duration

Social and economic: Emp.var.rate, Cons.price.idx, Cons.conf.idx, Euribor3m, Nr.employed

Other: Campaign, Pdays, Previous, Poutcome

## Results

Confusion Matrix: |7139 176|
                  |560  362|

Accuracy Score: 91

Area Under the Curve (AUC): 0.92

## Discussion
- Base on the results above, we can see that logistic regression is a good model (Accuracy score 91) to predict if the clients will subscribe a term deposit.
- Another way to evaluate the performance of the classifier is ROC curve. When the AUC is 1 it represents a perfect classification while 0.5 is a worthless classification, in our case AUC=0.92 which tells us that the performance of the classifier is pretty good.
- As for the confusion matrix we can see that the number of False Positives (560, ~7%) is pretty high, but since the aim of this classification problem is not to reach a life threatening decisions we can conclude that this model is a perfect fit for the type of problem we ae trying to find a solution for.
