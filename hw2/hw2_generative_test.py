import numpy as np
import pandas as pd 
from math import log, floor
from random import shuffle
import os, sys


    
def sigmoid(z):
    n = 1 / (1.0 + np.exp(-z))
    return np.clip(n, 1e-8, 1-(1e-8))
    
    
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
    df1 = pd.read_csv(argv[1], header=None, skiprows=1)
    df1.columns = [
        "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry"
    ]
    df1.isnull().values.any()
    df1.drop("Relationship", axis=1, inplace=True,)
    df1.drop("NativeCountry", axis=1, inplace=True,)
    df1.drop("Race", axis=1, inplace=True,)
    df1.drop("Education", axis=1, inplace=True,)
    df1.Age = df1.Age.astype(float)
    df1.fnlwgt = df1.fnlwgt.astype(float)
    df1.EducationNum = df1.EducationNum.astype(float)
    df1.HoursPerWeek = df1.HoursPerWeek.astype(float)
    df1.Age = (df1.Age-df.Age.mean())/df.Age.std()
    df1.fnlwgt = (df1.fnlwgt-df.fnlwgt.mean())/df.fnlwgt.std()
    df1.EducationNum = (df1.EducationNum-df.EducationNum.mean())/df.EducationNum.std()
    df1.HoursPerWeek = (df1.HoursPerWeek-df.HoursPerWeek.mean())/df.HoursPerWeek.std()
    df1.CapitalGain = (df1.CapitalGain-df1.CapitalGain.mean())/df1.CapitalGain.std()
    df1.CapitalLoss = (df1.CapitalLoss-df1.CapitalLoss.mean())/df1.CapitalLoss.std()
    df.Age = (df.Age-df.Age.mean())/df.Age.std()
    df.fnlwgt = (df.fnlwgt-df.fnlwgt.mean())/df.fnlwgt.std()
    df.EducationNum = (df.EducationNum-df.EducationNum.mean())/df.EducationNum.std()
    df.HoursPerWeek = (df.HoursPerWeek-df.HoursPerWeek.mean())/df.HoursPerWeek.std()
    df.CapitalGain = (df.CapitalGain-df.CapitalGain.mean())/df.CapitalGain.std()
    df.CapitalLoss = (df.CapitalLoss-df.CapitalLoss.mean())/df.CapitalLoss.std()
    df1 = pd.get_dummies(df1, columns=[
        "WorkClass", "MaritalStatus", "Occupation",
         "Gender",
    ])
    
    missing_cols = set( df.columns ) - set( df1.columns )
    for c in missing_cols:
        df1[c] = 0
    df1 = df1[df.columns]
    X_train = df.values
    y_train = y_all
    X_test = df1.values
    

    mu1 = np.loadtxt(os.path.join('parameter', 'm1'))
    mu2 = np.loadtxt(os.path.join('parameter', 'm2'))
    shared_sigma = np.loadtxt(os.path.join('parameter', 'shared_sigma'))
    N1 = np.loadtxt(os.path.join('parameter', 'N1'))
    N2 = np.loadtxt(os.path.join('parameter', 'N2'))

    
    sigma_inverse = shared_sigma
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_test.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)

    
    filename = argv[5]
    if not os.path.exists(filename):
        os.mkdir(filename)
    output_path = os.path.join(filename, 'prediction.csv')
    
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))
    
    return 0  
  
if __name__ == '__main__':  
    sys.exit(main(sys.argv[1:]))



