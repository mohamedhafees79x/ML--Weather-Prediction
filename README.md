# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages using import statement.
2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3. Import KMeans and use for loop to cluster the data.
4. Predict the cluster and plot data graphs.
5. Print the outputs and end the program

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Mohamed Hafees R
RegisterNumber:  212225230175
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data=pd.read_csv("Mall_Customers (1).csv")
data.head()
data.info()
data.isnull()
data.isnull().sum()

wcss= []

for i in range(1,11):
    kmeans=KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km=KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])

y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="black",label="clu
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="cyan",label="clust
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="yellow",label="cl
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="blue",label="clust
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="green",label="clu
plt.legend()
plt.title("Customer Segments")

```

## Output:

<h3>Data Head:</h3>
<img width="872" height="292" alt="image" src="https://github.com/user-attachments/assets/6a2cdb92-217b-4296-b2c4-d669ef3cf7db" />

<h3>Checking For Null Data:</h3>
<img width="405" height="175" alt="image" src="https://github.com/user-attachments/assets/93bc09de-2e96-46aa-8dcf-214823cb91be" />

<h3>Data information:</h3>
<img width="635" height="330" alt="image" src="https://github.com/user-attachments/assets/5dfe0036-79d7-43f4-98f7-6efbe1db7ce4" />

<h3>Within Cluster Sum of Square (WCSS):</h3>
<img width="868" height="283" alt="image" src="https://github.com/user-attachments/assets/7ead4d03-25d2-4833-af2f-bb137a673775" />

<h3>Elbow Method:</h3>
<img width="665" height="482" alt="image" src="https://github.com/user-attachments/assets/3c5acdf4-ab40-4aa5-88ac-f898bb9f56b4" />

<h3>K-means:<h3>
<img width="262" height="37" alt="image" src="https://github.com/user-attachments/assets/4867a867-c51e-4dec-9671-0b2bd78a8c1e" />

<h3>Cluster:</h3>
<img width="612" height="461" alt="image" src="https://github.com/user-attachments/assets/05172982-385c-4d90-8ccf-b6a0a82a6ac5" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
