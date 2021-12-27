import pandas as pd
import numpy as np
from scipy import stats
import os

def computeTandP(arr1,arr2):
    #t,p=0,0
    mean1,mean2=np.mean(arr1),np.mean(arr2)
    s1,s2=np.std(arr1),np.std(arr2)
    var=(((arr1.size-1)*(s1**2)) + ((arr2.size-1)*(s2**2))) / (arr1.size+arr2.size-2)
    sp=np.sqrt(var) #standard deviation
    ste=sp*np.sqrt((1/arr1.size) +(1/arr2.size)) #standard error
    t=(mean1-mean2)/ste
    df=arr1.size+arr2.size-2 #degrees of freedom
    p=2*(1-stats.t.cdf(t,df))
    return t,p

def removeOutliers(arr):
    q3,q1 = np.percentile(arr,[75,25])
    iqr=q3-q1
    lower=q1-1.5*iqr
    upper=q3+1.5*iqr
    arr=arr[(arr>=lower)&(arr<=upper)]
    return arr


# Read from the scores.csv using pandas
df=pd.read_csv('scores.csv')
#print(df.shape)
#print(df.head())
#print(df.columns)

fm=df.loc[df['gender']=='female','math score'].values
mm=df.loc[df['gender']=='male','math score'].values
#print(type(mm))
#print(fm)
#print(mm.size,",", fm.size)
fm=removeOutliers(fm)
mm=removeOutliers(mm)
t,p=computeTandP(mm,fm)
print("t value:",t)
print("p value:",p)
