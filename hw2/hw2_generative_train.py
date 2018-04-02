
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from math import log, floor
from random import shuffle
import os, sys



def sigmoid(z):
    n = 1 / (1.0 + np.exp(-z))
    return np.clip(n, 1e-8, 1-(1e-8))
    
def shuf(X_train, Y_train):
    random = np.arange(len(X_train))
    np.random.shuffle(random)
    return (X_train[random], Y_train[random])

def main(argv):     
    df = pd.read_csv(argv[0], header=None, skiprows=1)
    df.columns = [
        "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
    ]
    df.isnull().values.any()
    df.Income.unique()
    df["Income"] = df["Income"].map({ " <=50K": 0, " >50K": 1 })
    y_all = df["Income"].values
    df.drop("Income", axis=1, inplace=True,)
    df.drop("Relationship", axis=1, inplace=True,)
    df.drop("NativeCountry", axis=1, inplace=True,)
    df.drop("Race", axis=1, inplace=True,)
    df.drop("Education", axis=1, inplace=True,)
    df.Age = df.Age.astype(float)
    df.fnlwgt = df.fnlwgt.astype(float)
    df.EducationNum = df.EducationNum.astype(float)
    df.HoursPerWeek = df.HoursPerWeek.astype(float)
    df = pd.get_dummies(df, columns=[
        "WorkClass",  "MaritalStatus", "Occupation", 
         "Gender", 
    ])
    df.Age = (df.Age-df.Age.mean())/df.Age.std()
    df.fnlwgt = (df.fnlwgt-df.fnlwgt.mean())/df.fnlwgt.std()
    df.EducationNum = (df.EducationNum-df.EducationNum.mean())/df.EducationNum.std()
    df.HoursPerWeek = (df.HoursPerWeek-df.HoursPerWeek.mean())/df.HoursPerWeek.std()
    df.CapitalGain = (df.CapitalGain-df.CapitalGain.mean())/df.CapitalGain.std()
    df.CapitalLoss = (df.CapitalLoss-df.CapitalLoss.mean())/df.CapitalLoss.std()

    X_train = df.values
    y_train = y_all





    # In[8]:


    alldata_size = len(X_train)
    validdataset_size = int(floor(alldata_size * 0.1))# 10%

    X_all, Y_all = shuf(X_train, y_train)

    X_train, Y_train = X_all[0:validdataset_size], Y_all[0:validdataset_size]
    X_valid, Y_valid = X_all[validdataset_size:], Y_all[validdataset_size:]


    # In[9]:





    # In[10]:


    train_data_size = X_train.shape[0]
    c1 = 0
    c2 = 0

    m1 = np.zeros((39,))
    m2 = np.zeros((39,))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            m1 += X_train[i]
            c1 += 1
        else:
            m2 += X_train[i]
            c2 += 1
    m1 /= c1
    m2 /= c2

    sigma1 = np.zeros((39,39))
    sigma2 = np.zeros((39,39))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            sigma1 += np.dot(np.transpose([X_train[i] - m1]), [(X_train[i] - m1)])
        else:
            sigma2 += np.dot(np.transpose([X_train[i] - m2]), [(X_train[i] - m2)])
    sigma1 /= c1
    sigma2 /= c2
    shared_sigma = (float(c1) / train_data_size) * sigma1 + (float(c2) / train_data_size) * sigma2
    N1 = c1
    N2 = c2


    if not os.path.exists('parameter'):
        os.mkdir('parameter')
    param = {'m1':m1, 'm2':m2, 'shared_sigma':shared_sigma, 'N1':[N1], 'N2':[N2]}
    for key in sorted(param):
        
        np.savetxt(os.path.join('parameter', ('%s' % key)), param[key])



    sigma_inverse = shared_sigma 
    w = np.dot( (m1-m2), sigma_inverse)
    x = X_valid.T
    b = (-0.5) * np.dot(np.dot([m1], sigma_inverse), m1) + (0.5) * np.dot(np.dot([m2], sigma_inverse), m2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Valid AC = %f' % (float(result.sum()) / result.shape[0]))

if __name__ == '__main__':  
    sys.exit(main(sys.argv[1:]))


