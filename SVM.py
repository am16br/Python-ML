#Aidan Martin
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import sklearn

X_test = scipy.io.loadmat('data/data1/X_test.mat')['X_test']
X_train = scipy.io.loadmat('data/data1/X_train.mat')['X_train']
y_test = scipy.io.loadmat('data/data1/Y_test.mat')['y_test']
y_train = scipy.io.loadmat('data/data1/Y_train.mat')['y_train']

#model for each
svp=[SVC(kernel='poly',degree=2) for i in range(y_train.shape[1])]  #polynomial kernel
y_pred1=[]
svg=[SVC(kernel='rbf',degree=2) for i in range(y_train.shape[1])]   #Gaussian kernel
y_pred2=[]
for label in range(y_train.shape[1]):   #train SVM for each label/class
    svp[label].fit(X_train, y_train[:,label])   #polynomial kernel w/ parameter 2
    y_pred1.append(svp[label].predict(X_test))

    svg[label].fit(X_train, y_train[:,label])   #Gaussian kernel w/ parameter 2
    y_pred2.append(svg[label].predict(X_test))

a1,a2=[],[]
for sample in range(y_test.shape[0]):
    num,den,num1,den1=0,0,0,0
    for ind, label in enumerate(y_pred1):
        if(y_pred1[ind][sample] == 1 and y_test[sample][ind]==1):
            num+=1  #both have value 1
        if(y_pred1[ind][sample] == 1 or y_test[sample][ind]==1):
            den+=1  #at least one has value 1
        if(y_pred2[ind][sample] == 1 and y_test[sample][ind]==1):
            num1+=1 #repeated for predictions with other kernel
        if(y_pred2[ind][sample] == 1 or y_test[sample][ind]==1):
            den1+=1
    a1.append(num/den)
    a2.append(num1/den1)

print("Polynomial kernel accuracy= "+str(round(sum(a1)/len(a1),3)))
print("Gaussian kernel accuracy= "+str(round(sum(a2)/len(a2),3)))
