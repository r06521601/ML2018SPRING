import numpy as np
import scipy as sy
import pandas as pd
import os
import sys  

def standardize(X):
    
    m, n = X.shape
    
    for j in range(n):
        features = X[:,j]
        meanVal = features.mean(axis=0)
        std = features.std(axis=0)
        if std != 0:
            X[:, j] = (features-meanVal)/std
        else:
            X[:, j] = 0
    return X  
    
def main(argv):  
    test_x = []

    test_file_dir = argv[0]

    text = open(test_file_dir, 'r', encoding='big5')

    row = pd.read_csv(text,header=None)


    row = row.drop(row.columns[0:2], axis=1)

    n=0

    for i in range(0,row.shape[0]):
        if i %18 == 0:
            test_x.append([])
            
            for s in range(10,11):
                if row.iloc[i][s] != "NR":
                    test_x[n//18].append(float(row.iloc[i][s]))

                else:
                    test_x[n//18].append(float(0))
        else:
            
            for s in range(10,11):
                if row.iloc[i][s] != "NR":
                    test_x[n//18].append(float(row.iloc[i][s]))
                ###
                else:
                    test_x[n//18].append(float(0))    
        n = n+1
        
    text.close()
    test_x = np.array(test_x)  
    #test_x = standardize(test_x)    
    test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
    w = np.load('model.npy')
    ans = []
    for i in range(len(test_x)):
        ans.append(["id_"+str(i)])
        a = np.dot(w,test_x[i])
        ans[i].append(a)

    filename = argv[1]
    text = open(filename, "w+")


    df = pd.DataFrame(ans, columns=['id', 'value'])
    df.to_csv(text, sep=',',index = False)
    text.close()
    
    return 0  
  
if __name__ == '__main__':  
    sys.exit(main(sys.argv[1:]))



