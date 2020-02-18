#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import KFold
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces
# tells matplotlib to embed plots within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns  # higher level plotting tools
from sklearn.model_selection import KFold


# In[2]:


#data_frame2 = pd.read_csv(r'C:\Users\user\Desktop\Data\house_data_complete.csv')
data_frame1 = pd.read_csv(r'C:\Users\user\Desktop\Data\house_prices_data_training_data.csv')


# In[3]:


del data_frame1['date']


# In[4]:


del data_frame1['lat']


# In[5]:


del data_frame1['long']


# In[6]:


del data_frame1['id']


# In[7]:


del data_frame1['grade']


# In[8]:


del data_frame1['sqft_above']


# In[9]:


del data_frame1['sqft_living15']


# In[10]:


del data_frame1['sqft_lot15']


# In[11]:


data_frame1=data_frame1.dropna()


# In[12]:


data_frame_fold=data_frame1.copy()


# In[13]:


y=data_frame1.price


# In[14]:


print(y.shape[0])
print(data_frame1.shape[0])


# In[15]:


del data_frame1['price']


# In[16]:



X_train, X_test_val, y_train, y_test_val = train_test_split(data_frame1, y, test_size=0.4)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)


# In[17]:


print(y_train.shape[0])
print(X_train.shape[0])


# In[18]:


#print(X_test)


# In[19]:


#print(X_train.size)


# In[20]:


#print(X_val)


# In[21]:


def plotData(X, y,s,p):
    fig = pyplot.figure()
    pyplot.plot(X,y,'ro', ms=10, mec='k')
    pyplot.xlabel(s)
    pyplot.ylabel(p)
  


# In[22]:


#plotData(data_frame1['bedrooms'],data_frame1['price'], 'bedrooms','prices')


# In[23]:


sns.pairplot(data=data_frame_fold, x_vars=['sqft_living','sqft_lot','bedrooms','bathrooms','condition'], y_vars=["price"])


# In[24]:


data_frame1.isnull().values.any()


# In[25]:


#np.asarray(X_train)


# In[26]:


#np.asarray(y_train)


# In[27]:


#print(x_mean)


# In[28]:


#print(y_mean)


# In[29]:


#print(x_std)


# In[30]:


#print(y_std)


# In[31]:


#print(X_norm)


# In[32]:


def computeCostMulti(X, y, theta):
    
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    # You need to return the following variable correctly
    J = 0
    
    # ======================= YOUR CODE HERE ===========================
    
   
    J=(np.dot(((np.dot(X,theta)-y).T),(np.dot(X,theta)-y)))/2*m
    # ==================================================================
    return J


# In[33]:


def  featureNormalize(X):
 
 mu = np.mean(X);
 sigma = np.std(X);
 X_norm = (X - mu) / sigma;


 # ================================================================
 return X_norm


# In[34]:


def gradientDescentMulti(X, y, theta, alpha, num_iters): 
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()
    
    J_history = []
    
    for i in range(num_iters):
        # ======================= YOUR CODE HERE ==========================
       
        theta=theta-(alpha/m)*(np.dot(X,theta.T)-y).dot(X)
        
         # =================================================================
        
        # save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta))
    
    return theta, J_history


# In[35]:


alpha=0.3
num_iters=40
ones=np.ones(X_train.shape[0])

X1=featureNormalize(X_train)

theta=np.zeros(13)

X1['Xo']=ones
#print(X1)
print(y_train.size)
print(X1.shape[0])

h=X1.dot(theta)
theta, J_history=gradientDescentMulti(X1, y_train, theta, alpha, num_iters)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# Display the gradient descent's result
print('theta computed from gradient descent: {:s}'.format(str(theta)))


# In[36]:


alpha=0.3
#num_iters=40
ones=np.ones(X_val.shape[0])

X2=featureNormalize(X_val)

#theta=np.zeros(13)

X2['Xo']=ones


h=X2.dot(theta)
#theta, J_history=gradientDescentMulti(X2, y_val, theta, alpha, num_iters)

final1=computeCostMulti(X2, y_val, theta)

#Display the cost of cross validation 
print('cost of CV of H1: {:s}'.format(str(final1)))


# In[37]:


alpha=0.3

ones=np.ones(X_test.shape[0])

X3=featureNormalize(X_test)


X3['Xo']=ones


h=X3.dot(theta)


final2=computeCostMulti(X3, y_test, theta)

#Display the cost of cross validation 
print('cost of test of H1: {:s}'.format(str(final2)))


# In[38]:


alpha=0.3
num_iters=40
ones=np.ones(X_train.shape[0])

X4=featureNormalize(X_train)

theta1=np.zeros(13)

X4['Xo']=ones

X4['floors'] = np.square(X4['floors'])
#X4['bedrooms'] = np.square(X4['bedrooms'])
#X4['view'] = np.square(X4['view'])
#X4['sqft_living'] = np.square(X4['sqft_living'])



h= X4.dot(theta1)
theta1, J_history1=gradientDescentMulti(X4, y_train, theta1, alpha, num_iters)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history1)), J_history1, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# Display the gradient descent's result
print('theta computed from gradient descent: {:s}'.format(str(theta1)))


# In[39]:


alpha=0.3

ones=np.ones(X_val.shape[0])

X5=featureNormalize(X_val)



X5['Xo']=ones


h=X5.dot(theta1)

final3=computeCostMulti(X5, y_val, theta1)

#Display the cost of cross validation 
print('cost of CV of H2: {:s}'.format(str(final3)))


# In[40]:


alpha=0.3
ones=np.ones(X_test.shape[0])

X6=featureNormalize(X_test)

#theta=np.zeros(14)

X6['Xo']=ones


h=X6.dot(theta1)
#theta, J_history=gradientDescentMulti(X2, y_val, theta, alpha, num_iters)

final4=computeCostMulti(X6, y_test, theta1)

#Display the cost of cross validation 
print('cost of test of H2: {:s}'.format(str(final4)))


# In[41]:


alpha=0.3
num_iters=40
ones=np.ones(X_train.shape[0])

X7=featureNormalize(X_train)

theta2=np.zeros(13)

X7['Xo']=ones
X7['bathrooms'] = np.square(X7['bathrooms'])
#X6['sqft_living'] = np.square(X6['sqft_living'])
#print(y_train.size)
#print(X4.shape[0])

h= X7.dot(theta2)
theta2, J_history2=gradientDescentMulti(X7, y_train, theta2, alpha, num_iters)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history2)), J_history2, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# Display the gradient descent's result
print('theta computed from gradient descent: {:s}'.format(str(theta2)))


# In[42]:


alpha=0.3
#num_iters=40
ones=np.ones(X_val.shape[0])

X8=featureNormalize(X_val)

#theta=np.zeros(14)

X8['Xo']=ones


h=X8.dot(theta2)
#theta, J_history=gradientDescentMulti(X2, y_val, theta, alpha, num_iters)

final5=computeCostMulti(X8, y_val, theta2)


#Display the cost of cross validation 
print('cost of CV of H3: {:s}'.format(str(final5)))


# In[43]:


alpha=0.3
num_iters=40
ones=np.ones(X_test.shape[0])

X9=featureNormalize(X_test)


X9['Xo']=ones
X9['bathrooms'] = np.square(X9['bathrooms'])

h= X9.dot(theta2)

final6=computeCostMulti(X9, y_test, theta2)



#Display the cost of test
print('cost of test of H3: {:s}'.format(str(final6)))


# In[44]:


kf=KFold(n_splits=3,shuffle=True,random_state=2)
result=next(kf.split(data_frame_fold),None)
#print (result)
trainf=data_frame_fold.iloc[result[0]]
testf=data_frame_fold.iloc[result[1]]
#print(trainf)

y_train_fold=trainf.price
y_test_fold=testf.price
#print (y_train_fold)
del trainf['price']
del testf['price']
X_trainf=trainf
X_testf=testf
#print(trainf)

alpha=0.3
num_iters=40

ones=np.ones(X_trainf.shape[0])

X10=featureNormalize(X_trainf)

theta3=np.zeros(13)

X10['Xo']=ones


h=X10.dot(theta3)
theta3, J_history5=gradientDescentMulti(X10, y_train_fold, theta3, alpha, num_iters)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history5)), J_history5, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# Display the gradient descent's result
print('theta computed from gradient descent: {:s}'.format(str(theta3)))


# In[45]:


alpha=0.3
#num_iters=40
ones=np.ones(X_testf.shape[0])

X10=featureNormalize(X_testf)

#theta=np.zeros(14)

X10['Xo']=ones


h=X10.dot(theta3)
#theta, J_history=gradientDescentMulti(X2, y_val, theta, alpha, num_iters)

final7=computeCostMulti(X10, y_test_fold, theta3)


#Display the cost of cross validation 
print('cost of test of H1: {:s}'.format(str(final7)))

