# NEURAL NETWORK

Neural networks are a type of machine learning learning technique that is loosely based on the concept of biological neural networks. In general, neural networks are self-optimizing functions that translate inputs to the expected output. The function can then take a fresh input and estimate an output based on the function it established with the dataset. The key metrics of Neural Networks are :
**Accuracy:** Accuracy defines the ratio of exact predictions to the total predications made which is equal to 95%. This is evaluated as (TP+TN)/(TP+TN+FP+FN)
**Precision:** Precision defines how accurate the model is and it is equal to 92%. This is calculated as TP/(TP+FP).
**Recall:** Recall defines the ratio of exactly predicted positive to all the observations and its is equal to 94%. This is calculated as TP/(TP+FN).
**F1:** F1 is a metric for determining how accurate a test is. It is determined using the test's precision and recall, where the precision is the number of correctly identified positive results divided by the total number of positive results.

## OBJECTIVE

Here we are going to compare the key metrics of Neural Networks Algorithm to Logistical Regression Algorithm to see their accuracy in predicting the output.

## PREREQUISITES

We need to install the Anaconda and Python from google in order to carry out this analysis.
```
- Anaconda3-2021.05-Windows-x86_64
- python.exe
```

## INSTALLATION

Download the Anaconda3-2021.05-Windows-x86_64 file from Google.
Right click and Run as admin.
Click on I agree to accept the terms of agreement.
Select a path to store the data and click NEXT.
Let it install all the packages and once it shows installation complete, click on FINISH.

Download python.exe from Google.
Right click and Run as admin.
A black screen of cmd pops up and runs the installation.

## RUNNING THE TESTS

After installation is completed, open Anaconda and run the 'base(root)' environment.
Install Jupyter Notebook and launch.

## BREAK DOWN INTO END TO END TESTS

Click and upload a .csv file.

```
drugdataset.csv
```

## ADD CODING STYLE TESTS

Data insights are the deep understanding that an individual or organisation gains through analysing data on a certain topic. Organizations can make better decisions based on this in-depth information than they could if they relied just on intuition.
Key statistics are totals and statistics derived from data found in any dataset. This contains things like Mean, Standard Deviation, and Correlation, among other things.
```
#Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#Load Data
data = pd.read_csv('./drugdataset.csv')
data.head()
Age	Sex	BP	Cholesterol	Na_to_K	Drug
0	23	1	2	1	25.355	drugY
1	47	0	1	1	13.093	drugC
2	47	0	1	1	10.114	drugC
3	28	1	0	1	7.798	drugX
4	61	1	1	1	18.043	drugY

#Identify number of Classes (i.e. Species)
data.Drug.unique()
array(['drugY', 'drugC', 'drugX', 'drugA', 'drugB'], dtype=object)
#Create x and y variables
X = data.drop('Drug',axis=1).to_numpy()
y = data['Drug'].to_numpy()

#Create Train and Test datasets
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size = 0.2,random_state=100)

#Scale the data
from sklearn.preprocessing import StandardScaler  
sc = StandardScaler()  
x_train2 = sc.fit_transform(X_train)
x_test2 = sc.transform(X_test)
#Script for Neural Network
from sklearn.neural_network import MLPClassifier  
mlp = MLPClassifier(hidden_layer_sizes=(5,4,5),
                    activation='relu',solver='adam',
                    max_iter=10000,random_state=100)  
mlp.fit(x_train2, y_train) 
predictions = mlp.predict(x_test2) 

#Evaluation Report and Matrix
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions)) 
[[ 5  0  0  0  0]
 [ 0  2  0  0  1]
 [ 0  0  3  0  0]
 [ 0  0  0 11  0]
 [ 0  0  0  1 17]]
              precision    recall  f1-score   support

       drugA       1.00      1.00      1.00         5
       drugB       1.00      0.67      0.80         3
       drugC       1.00      1.00      1.00         3
       drugX       0.92      1.00      0.96        11
       drugY       0.94      0.94      0.94        18

    accuracy                           0.95        40
   macro avg       0.97      0.92      0.94        40
weighted avg       0.95      0.95      0.95        40

#Script for Decision Tree
from sklearn.tree import DecisionTreeClassifier  

for name,method in [('DT', DecisionTreeClassifier(random_state=100))]: 
    method.fit(x_train2,y_train)
    predict = method.predict(x_test2)
    print('\nEstimator: {}'.format(name)) 
    print(confusion_matrix(y_test,predict))  
    print(classification_report(y_test,predict))    
Estimator: DT
[[ 4  1  0  0  0]
 [ 0  3  0  0  0]
 [ 0  0  3  0  0]
 [ 0  0  0 10  1]
 [ 0  0  0  0 18]]
              precision    recall  f1-score   support

       drugA       1.00      0.80      0.89         5
       drugB       0.75      1.00      0.86         3
       drugC       1.00      1.00      1.00         3
       drugX       1.00      0.91      0.95        11
       drugY       0.95      1.00      0.97        18

    accuracy                           0.95        40
   macro avg       0.94      0.94      0.93        40
weighted avg       0.96      0.95      0.95        40

#To determine the statistical value
data.describe()
Age	Sex	BP	Cholesterol	Na_to_K
count	200.000000	200.000000	200.000000	200.000000	200.000000
mean	44.315000	0.480000	1.090000	0.515000	16.084485
std	16.544315	0.500854	0.821752	0.501029	7.223956
min	15.000000	0.000000	0.000000	0.000000	6.269000
25%	31.000000	0.000000	0.000000	0.000000	10.445500
50%	45.000000	0.000000	1.000000	1.000000	13.936500
75%	58.000000	1.000000	2.000000	1.000000	19.380000
max	74.000000	1.000000	2.000000	1.000000	38.247000

#To determine the correlation
data.corr()
Age	Sex	BP	Cholesterol	Na_to_K
Age	1.000000	-0.102027	-0.054212	0.068234	-0.063119
Sex	-0.102027	1.000000	-0.007814	-0.008811	0.125008
BP	-0.054212	-0.007814	1.000000	-0.137552	0.149312
Cholesterol	0.068234	-0.008811	-0.137552	1.000000	-0.010000
Na_to_K	-0.063119	0.125008	0.149312	-0.010000	1.000000

```

## REAL-TIME USE

They can be used to detect patterns in data or to represent complex interactions between inputs and outputs. In the process of data mining, data warehousing companies use neural networks as a method to extract information from databases.

## ABOUT THE SOFTWARE

Anaconda is a Python and R programming language distribution aimed for simplifying package management and deployment in scientific computing (data science, machine learning applications, large-scale data processing, predictive analytics, and so on).
```
Developer - Anaconda, Inc. (previously Continuum Analytics)
Initial release	- 0.8.0/17 July 2012; 9 years ago
Stable release - 2021.05 / 13 May 2021; 6 months ago
Written in	Python
Operating system - Windows, macOS, Linux
Type - Programming language, machine learning, data science
License - Freemium (Miniconda and the Individual Edition are free software, but the other editions are software as a service)
Website - anaconda.com
```
## VERSION

Anaconda's package management system, conda, keeps track of package versions. This package manager was spun off as a distinct open-source package because it turned out to be valuable in and of itself, not just for Python.

## AUTHOR

Guido van Rossum - Guido van Rossum is a well-known Dutch programmer who is best known for inventing the Python programming language.

## LICENSE

Anaconda is licensed under Freemium (Miniconda and the Individual Edition are free software, but the other editions are software as a service)

## CONCLUSION

As a result of the findings, we may conclude that the neural network method produces correct values for the provided dataset.
