#Aidan Martin
import scipy.io
from sklearn.linear_model import LogisticRegression
import numpy as np
from random import sample
import matplotlib.pyplot as plt

def load(folder, dataset, setNum):
    testing_labels = scipy.io.loadmat(folder+'testingLabels'+dataset+setNum+'.mat')['testingLabels']
    testing_matrix = scipy.io.loadmat(folder+'testingMatrix'+dataset+setNum+'.mat')['testingMatrix']
    if(dataset=='_MindReading'):
        training_labels = scipy.io.loadmat(folder+'trainingLabels'+dataset+'_'+setNum+'.mat')['trainingLabels']  #c'mon
        unlabeled_labels = scipy.io.loadmat(folder+'unlabeledLabels'+dataset+'_'+setNum+'.mat')['unlabeledLabels']  #rusrius
    else:
        training_labels = scipy.io.loadmat(folder+'trainingLabels'+dataset+setNum+'.mat')['trainingLabels']
        unlabeled_labels = scipy.io.loadmat(folder+'unlabeledLabels'+dataset+setNum+'.mat')['unlabeledLabels']
    training_matrix = scipy.io.loadmat(folder+'trainingMatrix'+dataset+setNum+'.mat')['trainingMatrix']
    unlabeled_matrix = scipy.io.loadmat(folder+'unlabeledMatrix'+dataset+setNum+'.mat')['unlabeledMatrix']
    return testing_labels, testing_matrix, training_labels, training_matrix, unlabeled_labels, unlabeled_matrix

def random_sampling(set,k): #Select a batch of k samples from the unlabeled set at random
    return sample(range(len(set)),k)    #return random indicies

def uncertainty_sampling(set,k): #select top-k samples with highest classification entropy
    e=(-set*np.log2(set)).sum(axis=1)   #apply entropy function[e = - ∑ pi log (pi)] on each sample (each sample is row: array of probabilities sample belongs to each class)
    return np.argpartition(e, -k)[-k:]  #return indicies of the k samples producing the highest entropy.

def LR_active_learning(N,k,strategy,testing_labels, testing_matrix, training_labels, training_matrix, unlabeled_labels, unlabeled_matrix):
    scores=[]   #will hold N accuracy values, 1 for each iteration
    for iter in range(N):   #Loop over N iterations
        #Step 1: Train a machine learning model using the current training set
        logisticRegr = LogisticRegression(solver='newton-cg')
        logisticRegr.fit(training_matrix, training_labels.ravel())
        #Step 2: Apply the model on the test set and obtain the accuracy
        predictions = logisticRegr.predict(testing_matrix)      #Make predictions on entire test data
        score = logisticRegr.score(testing_matrix, testing_labels.ravel())
        scores.append(score)
        #Step 3: Apply the model on the unlabeled set and select a batch of k unlabeled samples based on an active learning strategy (see below)
        #predictions = logisticRegr.predict(unlabeled_matrix)    #Predict class labels for samples in X.
        #score = logisticRegr.score(unlabeled_matrix, unlabeled_labels)
        if(strategy=='random'):
            selected=random_sampling(unlabeled_matrix, k)   #returns list of indices
        else:
            predictionsProb=logisticRegr.predict_proba(unlabeled_matrix)           #Probability estimates.
            selected=uncertainty_sampling(predictionsProb, k)
        #Step 4: Obtain the labels of the selected k samples from a human expert (you will use the provided labels of the unlabeled samples here to simulate the human expert)
        training_labels = np.vstack([training_labels, unlabeled_labels[selected]])
        #Step 5: Remove those samples from the unlabeled set and add them to the current training set
        training_matrix = np.vstack([training_matrix, unlabeled_matrix[selected]])
        unlabeled_labels = np.delete(unlabeled_labels, selected, axis=0)
        unlabeled_matrix = np.delete(unlabeled_matrix, selected, axis=0)
        #End Loop
    return scores

def plot(line1,line1label,line2,line2label,xlabel,ylabel,title):
    plt.plot(line1, label=line1label)
    plt.plot(line2, label=line2label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

folders=['data/MindReading/', 'data/MMI/']
datasets=['_MindReading','_']
k=10    #In this project, use k = 10
N=50    #and N = 50.
dataset_scores=[]
for each in range(2):          #Also, for each dataset,
    rScores,uScores=[],[]
    for setNum in range(3):    #repeat process 3 times using 3 different initial training sets, unlabeled sets and test sets and report the average accuracy results.
        teL, teM, trL, trM, unL, unM = load(folders[each],datasets[each], str(setNum+1))    #load dataset
        scores=LR_active_learning(N,k,'random',teL, teM, trL, trM, unL, unM)    #run strategy; return N accuracy scores
        rScores.append(scores)
        #repeat process with other active learning strategy
        teL, teM, trL, trM, unL, unM = load(folders[each],datasets[each], str(setNum+1))
        scores=LR_active_learning(N,k,'uncertainty',teL, teM, trL, trM, unL, unM)
        uScores.append(scores)
    #average over the three runs comparing Random and Uncertain... separate for each dataset
    dataset_scores.append([np.average(np.array(rScores), axis=0),np.average(np.array(uScores), axis=0)])
#For each dataset, plot one graph containing the performance of Random Sampling and Uncertainty-based Sampling (on the same graph), averaged over the three runs. Write your observations in your report.
plot(dataset_scores[0][0],"Random Sampling",dataset_scores[0][1],"Uncertainty-based Sampling","Iterations","Accuracy (%)","MindReading")
plot(dataset_scores[1][0],"Random Sampling",dataset_scores[1][1],"Uncertainty-based Sampling","Iterations","Accuracy (%)","MMI")
