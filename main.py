#Import the necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import random

#Load the data
data = load_breast_cancer()

#Print the data
print(data.keys())
#print the data features names
print(data['feature_names'])

#print the data target names
print(data['target_names'])

#X is the data and y is the target
X = data['data']
y = data['target']

#Split the data into training and testing data with 20% of the data for testing and 80% for training also set the random state to 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create the classifier with the KNeighborsClassifier algorithm
clf = KNeighborsClassifier()

#Fit the classifier to the data
clf.fit(X_train, y_train)

#print the score + the confusion matrix + the classification report
print(clf.score(X_test, y_test))

#Predict new data
print(len(data['feature_names']))
X_new = np.array(random.sample(range(0,50), 30))
print(data['target_names'][clf.predict([X_new])[0]])

##############################Let's do more!#####################################
#Import the necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Concatenate the data and the target in one array
column_data = np.concatenate([data['data'], data['target'][:,None]], axis=1)
print(column_data)

#Create a dataframe with the data and the target class
column_names = np.concatenate([data['feature_names'], ['class']])
df = pd.DataFrame(column_data, columns=column_names)
print(df)

#Print the correlation between the features: Correlation is a statistical measure that indicates the extent to which two or more variables fluctuate together.
print(df.corr())

#Plot the correlation using the heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', annot_kws={'fontsize': 8})
plt.show()

