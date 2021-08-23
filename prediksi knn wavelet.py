import os
from os.path import dirname, join as pjoin
import numpy as np
import numpy as fitur
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pywt, math
import pylab
from tabulate import tabulate
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
data_dir = pjoin(dirname('C:\data'))
files = os.listdir('c:\data')
#print(files)
A=np.array([])
X=np.array([])
i=0;
head = ["std", "mean", "median", "min", "max"]
for file in files:
    i=i+1
    wav_fname = pjoin('C:\data', file)
    if (file[0]=='n'):
        #print("NORMAL")
        X=np.append(X,0)
    else:
        #print("WHEEZE")
        X=np.append(X,1)
    samplerate, data = wavfile.read(wav_fname)
    cA, cD = pywt.dwt(data, 'db4') #ganti untuk wavelet type lain
    #print(np.std(cA))
    # print(np.mean(cA))
    #print(np.min(cA))
    # print(np.max(cA))
    #print(np.median(cA))
    A=np.append(A,[np.std(cA),np.mean(cA),np.median(cA),np.min(cA),np.max(cA)])
    if i==1:
        a=np.array([np.std(cA),np.mean(cA),np.median(cA),np.min(cA),np.max(cA)])
#print(A)
#print(i)
np.savetxt("data_fitur.csv", A, delimiter=",")
b=A.reshape(-1, 5)
c=a.reshape(-1, 5)
print(tabulate(b, headers=head, tablefmt="grid"))
#print(X)   
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(b,X)
d=knn.predict(c)
 

x=knn.predict(b)
print("DATA :");
print(X)
print("PREDIKSI :");
print(x)
#print(len(x))
TP=0
TN=0
FP=0
FN=0
for i in  range(0, len(x)):
 
    if X[i]==0:
        if x[i]==0:
            TN=TN+1
        else:
            FN=FN+1
    else:
        if x[i]==1:
            TP=TP+1
        else:
            FP=FP+1        
print("AKURASI :")
print((TP+TN)/(TP+TN+FP+FN)*100," %")
