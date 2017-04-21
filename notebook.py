import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import (RandomTreesEmbedding, IsolationForest, RandomForestClassifier, GradientBoostingClassifier)
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
import pickle
from sklearn import metrics
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np 
import json 

dataPrefix = '/home/vagrant/data/aaia17/'
def writeToFile(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write("%s\n" % item)
    f.close()
def printify(t):
    for x in t:
        print(x)
def readDataAsFrame(fNames):
    frame = []
    for fileName in fNames:
        fData = dataPrefix+fileName
        frame.append(pandas.read_csv(fData))
    return pandas.concat(frame)        
def getNewForIndexes(dataFrame, names):
    X_selectedFeature = np.empty((len(dataFrame), 0), float)
    for name in names:
        X_selectedFeature = np.column_stack([X_selectedFeature, dataFrame[name].values])
    return X_selectedFeature
def dataScaling(data, scaler):
    data = data/(data+1)
    return scaler.fit_transform(data)
dataPrefix = '/home/vagrant/data/aaia17/'
def writeToFile(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write("%s\n" % item)
    f.close()
def printify(t):
    for x in t:
        print(x)
def readDataAsFrame(fNames):
    frame = []
    for fileName in fNames:
        fData = dataPrefix+fileName
        frame.append(pandas.read_csv(fData))
    return pandas.concat(frame)        
def getNewForIndexes(dataFrame, names):
    X_selectedFeature = np.empty((len(dataFrame), 0), float)
    for name in names:
        X_selectedFeature = np.column_stack([X_selectedFeature, dataFrame[name].values])
    return X_selectedFeature
def dataScaling(data, scaler):
    data = data/(data+1)
    return scaler.fit_transform(data)

def read_sparse_binary_set(matrix_path):
    return np.load(matrix_path)




trainingFiles = ['trainingData_tabular_chunk1.csv', 'trainingData_tabular_chunk2.csv',
                 'trainingData_tabular_chunk3.csv', 'trainingData_tabular_chunk4.csv']
rawTrainingDataset = readDataAsFrame(trainingFiles)

testFiles = ['testData_tabular.csv']
rawTestDataset = readDataAsFrame(testFiles)


YRawTrainingDataset = rawTrainingDataset.values[:,1]
XRawTrainingDataset = rawTrainingDataset.values[:,2:45]
XRawTestDataset = rawTestDataset.values[:,2:45]

training_sparse = read_sparse_binary_set(dataPrefix + '/output/m.npy')
test_sparse = read_sparse_binary_set(dataPrefix + '/output/m_test.npy')

XRawTestDataset = np.concatenate((XRawTestDataset, test_sparse[:,0:78]), axis = 1)
XRawTrainingDataset = np.concatenate((XRawTrainingDataset, training_sparse[:,0:78]), axis = 1)
assert np.all(training_sparse[:,78] == rawTrainingDataset.values[:,0]) and  np.all(test_sparse[:,78] == rawTestDataset.values[:,0])


min_max_scaler = preprocessing.MinMaxScaler()

XTrainingDatasetScaledWithOutliers = dataScaling(XRawTrainingDataset, min_max_scaler)
XTestDatasetScaled = dataScaling(XRawTestDataset, min_max_scaler)
validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(XTrainingDatasetScaledWithOutliers, 
        YRawTrainingDataset, test_size=validation_size, random_state=seed)



import gc 
del training_sparse
del test_sparse
del rawTrainingDataset
del rawTestDataset
gc.collect()



clf = GradientBoostingClassifier(n_estimators=20, max_depth=4)
clf = clf.fit(XTrainingDatasetScaledWithOutliers, YRawTrainingDataset)
predictedTestWon = clf.predict_proba(XTestDatasetScaled)[:,1]
writeToFile(predictedTestWon, dataPrefix+'predicted22.txt')

