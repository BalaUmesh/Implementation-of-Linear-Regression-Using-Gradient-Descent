# Implementation-of-Linear-Regression-Using-Gradient-Descent
date: 7/9/23

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:

Program to implement the linear regression using gradient descent.
Developed by: Bala Umesh
RegisterNumber:  212221040024
```py
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X,y,theta and generate the cost function in a linear regression model

  """
  m=len(y)  
  h=X.dot(theta)
  square_err=(h - y)**2
  return 1/(2*m) * np.sum(square_err)

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  """
   Take in numpy array X,y and theta and update theta by taking number with learning rate of alpha

  return theta and the list of the cost of theta during each iteration
  """
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))

```

## Output:
![image](https://github.com/BalaUmesh/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113031742/35334cb3-6460-4f1e-bd3a-a85b64ab786f)
### profit prediction graph:
![image](https://github.com/BalaUmesh/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113031742/bee2b385-ef01-4bf6-9ba1-d889dd912764)


### compute cost value:
![image](https://github.com/BalaUmesh/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113031742/1489318c-88a9-4b7f-b866-448e902a2d3d)


### h(x) value:
![image](https://github.com/BalaUmesh/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113031742/15afbe50-6518-438e-b1d0-895b94cd5ea8)


### cost function using gradient descent graph:
![image](https://github.com/BalaUmesh/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113031742/fb6fcc81-1442-40d5-a5d5-9b87c165d868)


### profit prediction graph:
![image](https://github.com/BalaUmesh/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113031742/d5c196e5-e30a-4602-b887-5b0a6e41e034)


### profit for the population 35,000:
A![image](https://github.com/BalaUmesh/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113031742/a0bcdb3c-71d1-460a-885c-825d91a25c9c)


### profit for the population 70,000:
![image](https://github.com/BalaUmesh/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113031742/27e76b41-60a8-494b-8ec5-59abdd6a45dc)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
