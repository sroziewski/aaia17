from multiprocessing import Pool, Manager
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
import json
import numpy as np

def replaceNan(data, item):
    return [x if x is not None else item for x in data]

def getDenseFeatures(dataFrame, names, dictionary):
    X_dense = np.empty((len(dataFrame), 0), float)
    for name in names:
        values = dataFrame[name].values
        X_dense = np.column_stack([X_dense, replaceNan([dictionary[name].get(value) for value in values], 0.5)])
    return X_dense
def addNewFeature(data, vectors):
    return np.column_stack([data, vectors])
def getNewForIndexes(dataFrame, names):
    X_selectedFeature = np.empty((len(dataFrame), 0), float)
    for name in names:
        X_selectedFeature = np.column_stack([X_selectedFeature, dataFrame[name].values])
    return X_selectedFeature
def dataScaling(data, scaler):
    data = data/(data+1)
    return scaler.fit_transform(data)

def _computeFractionForAttribute(dataFrame, name):
    dictionary = dict()
    values = set(dataFrame[name].values)
    for value in values:
        ind = dataFrame[name].values==value
        subset = dataFrame['decision'][ind].values
        fraction = 1.0*len(filter(lambda x: x==1, subset)) / len(subset)
        dictionary[value] = fraction 
    return dictionary
def computeFractionsForAll(dataFrame, names):
    changeDict = dict()
    for name in names:
        changeDict[name] = _computeFractionForAttribute(dataFrame, name)
    return changeDict
def dropColumnsFromDataFrame(df, columns):
    for column in columns:
        df = df.drop(column, 1)
    return df

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

def read_sparse_binary_set(matrix_path):
    return np.load(matrix_path)

dataPrefix = '/home/data/aaia17/'
dataDepreciatedPrefix = dataPrefix+"depreciated/"
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
def readJsonData(fNames):
    frame = []
    for fileName in fNames:
        fData = dataPrefix+fileName
        with open(fData, 'r') as f:
            frame.append(json.load(f))
        f.close()
    return frame  
def readJsonDatabyLine(prefix, fNames):
    frame = []
    for fileName in fNames:
        fData = prefix+fileName
        frame.extend(readFileByLine(fData))
    return frame
def readJsonFromString(data):
    result = []
    nulls = []
    for i in range(0,len(data)):
        result.append(json.loads(data[i]))
                    #nulls.append(i)
    return result
def getNewForIndexes(dataFrame, names):
    X_selectedFeature = np.empty((len(dataFrame), 0), float)
    for name in names:
        X_selectedFeature = np.column_stack([X_selectedFeature, dataFrame[name].values])
    return X_selectedFeature

def read_sparse_binary_set(matrix_path):
    return np.load(matrix_path)
def readFileByLine(fileName):
    arr = []
    with open(fileName, 'r') as f:        
        for line in f:
            arr.append(line.rstrip())
        f.close()
    return arr

def getHandHpHistogram(state):
    handHp = np.zeros(3)
    # maxval is 8, we divide by 3
    for hand in state['player']['hand']:
        try:
            handHp[getBin(hand['hp'], 8, 3)] +=1
        except KeyError:
            pass
    return handHp
def getHandDurability(state):
    handDur = np.zeros(3)
    # maxval is 4, we divide by 3
    for hand in state['player']['hand']:
        if hand['type'] == "WEAPON":
            try:
                handDur[getBin(hand['durability'], 4, 3)] +=1
            except KeyError:
                pass
    return handDur
def getHandAttackHistogram(state, type):
    # maxval is 8, we divide by 3 folds
    maxval = 8
    div = 3
    handAttack = np.zeros(div)
    if type == "WEAPON":
        # maxval is 5, we divide by 3
        maxval = 5
        handAttack = np.zeros(div)
    for hand in state['player']['hand']:
        if hand['type'] == type:
            try:
                handAttack[getBin(hand['attack'], maxval, div)] +=1
            except KeyError:
                pass
    return handAttack
def getHandCrystalCostHistogram(state, type):
    # maxval is 8
    maxval = 8
    div = 3
    handCrystalCost = np.zeros(div) # we divide to 4 folds
    
    if type == "WEAPON":
        # maxval is 5
        handCrystalCost = np.zeros(div) # we divide to 3 folds
        maxval = 5
    if type == "SPELL":
        # maxval is 7
        handCrystalCost = np.zeros(div) # we divide to 3 folds
        maxval = 7
    for hand in state['player']['hand']:
        if hand['type'] == type:
            try:
                handCrystalCost[getBin(hand['crystals_cost'], maxval, div)] +=1
            except KeyError:
                pass
    return handCrystalCost

def getHandSumBySpecific(state, attr, specific):
    sum = 0
    for hand in state['player']['hand']:
        try:
            if hand[specific] is True and hand['type'] == "MINION":
                sum+=hand[attr]
        except KeyError:
            pass
    return sum

def getPlayedCardsSumBySpecific(state, attr, specific, type):
    sum = 0
    for hand in state[type]['played_cards']:
        try:
            if hand[specific] is True:
                sum+=hand[attr]
        except KeyError:
            pass
    return sum

def getPlayedCardsAllSpecifics(state, type):
    attrs = ['hp_max', 'hp_current', 'attack', 'crystals_cost']
    specifs = ['taunt', 'charge', 'freezing', 'frozen', 'can_attack']
    specs = np.zeros(20)
    i = 0
    for specif in specifs:
        for attr in attrs:
            specs[i] = getPlayedCardsSumBySpecific(state, attr, specif, type)
            i+=1
    return specs

def getHandAllSpecifics(state):
    attrs = ['hp', 'attack', 'crystals_cost']
    specifs = ['taunt', 'charge', 'freezing']
    specs = np.zeros(9)
    i = 0
    for specif in specifs:
        for attr in attrs:
            specs[i] = getHandSumBySpecific(state, attr, specif)
            i+=1
    return specs

def getHandCountsOfTypes(state):
    counts = np.zeros(3)
    for hand in state['player']['hand']:
        try:
            if hand['type']=="MINION":
                counts[0]+=1
            elif hand['type']=="WEAPON":
                counts[1]+=1    
            elif hand['type']=="SPELL":
                counts[2]+=1    
        except KeyError:
            pass
    return counts

def getBin(val, max, nb):
    if val > max:
        val = max
    dx = max/nb
    if max % nb >0:
        dx+=1
    r = val/dx
    if val == max and max % nb == 0:
        r-=1
    return r

def getPlayerStats(state, player):
    stats = np.zeros(6)
    i = 0
    data = state[player]['stats']
    stats[0] = data['crystals_all']
    stats[1] = data['crystals_current']
    stats[2] = data['deck_count']
    stats[3] = data['fatigue_damage']
    stats[4] = data['hand_count']
    stats[5] = data['played_minions_count']
    
    return stats

def getHeroStats(state, player):
    stats = np.zeros(5)
    data = state[player]['hero']
    stats[0] = data['armor']
    stats[1] = data['attack']
    stats[2] = data['hp']
    stats[3] = data['special_skill_used'] == True
    stats[4] = data['weapon_durability']
    
    return stats

def getPlayerPlayedCardsHpCurrentHistogram(state):
    #maxval is 24, we divide to 4 folds
    handHp = np.zeros(4)
    for hand in state['player']['played_cards']:
        try:
            handHp[getBin(hand['hp_current'], 24, 4)] +=1
        except KeyError:
            pass
    return handHp
def getOpponentPlayedCardsHpCurrentHistogram(state):
    #maxval is 32, we divide to 5 folds
    handHp = np.zeros(5)
    for hand in state['opponent']['played_cards']:
        try:
            handHp[getBin(hand['hp_current'], 32, 5)] +=1
        except KeyError:
            pass
    return handHp
def getPlayerPlayedCardsHpMaxHistogram(state):
    #maxval is 30, we divide to 3 folds
    handHp = np.zeros(3)
    for hand in state['player']['played_cards']:
        try:
            handHp[getBin(hand['hp_max'], 30, 3)] +=1
        except KeyError:
            pass
    return handHp
def getOpponentPlayedCardsHpMaxHistogram(state):
    #maxval is 37, we divide to 4 folds
    handHp = np.zeros(4)
    for hand in state['opponent']['played_cards']:
        try:
            handHp[getBin(hand['hp_max'], 37, 4)] +=1
        except KeyError:
            pass
    return handHp
def getPlayerPlayedCardsAttackHistogram(state):
    #maxval is 23, we divide to 4 folds
    handHp = np.zeros(4)
    for hand in state['player']['played_cards']:
        try:
            handHp[getBin(hand['attack'], 23, 4)] +=1
        except KeyError:
            pass
    return handHp
def getOpponentPlayedCardsAttackHistogram(state):
    #maxval is 19, we divide to 3 folds
    handHp = np.zeros(3)
    for hand in state['opponent']['played_cards']:
        try:
            handHp[getBin(hand['attack'], 19, 3)] +=1
        except KeyError:
            pass
    return handHp
def getPlayedCardsCrystalsCostHistogram(state, type):
    #maxval is 8, we divide to 3 folds
    handHp = np.zeros(3)
    for hand in state[type]['played_cards']:
        try:
            handHp[getBin(hand['crystals_cost'], 8, 3)] +=1
        except KeyError:
            pass
    return handHp

def craftAllFeatures(state):
    return np.hstack([
            getHandHpHistogram(state), getHandDurability(state), getHandAttackHistogram(state, "MINION"),
            getHandAttackHistogram(state, "WEAPON"), getHandCrystalCostHistogram(state, "MINION"),
            getHandCrystalCostHistogram(state, "WEAPON"), getHandCrystalCostHistogram(state, "SPELL"),
            getHandAllSpecifics(state), getPlayedCardsAllSpecifics(state, 'player'), getHandCountsOfTypes(state),
            getPlayerStats(state, 'player'), getHeroStats(state, 'player'), 
            getPlayerPlayedCardsHpCurrentHistogram(state), getPlayerPlayedCardsHpMaxHistogram(state),
            getPlayerPlayedCardsAttackHistogram(state), getPlayedCardsCrystalsCostHistogram(state, 'player'),
            getOpponentPlayedCardsHpCurrentHistogram(state), getOpponentPlayedCardsHpMaxHistogram(state),
            getOpponentPlayedCardsAttackHistogram(state), getPlayedCardsCrystalsCostHistogram(state, 'opponent'),
            getPlayedCardsAllSpecifics(state, 'opponent')
        ])

dataFeaturesOutput = '/home/data/aaia17/features/'

def getAllJsonTraining(data, id):
    handsHist = np.empty((0, 113), float)
    for state in data:
        handsHist = np.vstack([handsHist, craftAllFeatures(state)])
    pickle.dump(handsHist, open(dataFeaturesOutput+'handsHist-%d.store'%id,"wb"))
    #return handsHist

dtrain = readJsonDatabyLine(dataPrefix, ["trainingData_JSON_chunk1.json", "trainingData_JSON_chunk2.json", 
                                                "trainingData_JSON_chunk3.json", "trainingData_JSON_chunk4.json"])
dtest = readJsonDatabyLine(dataDepreciatedPrefix, ["testData_JSON_chunk5.json", "testData_JSON_chunk6.json", 
                                                "testData_JSON_chunk7.json"])
jtrain = readJsonFromString(dtrain)
jtest = readJsonFromString(dtest)
jtrainData = np.hstack([jtrain, jtest])

manager = Manager()
shared_list = manager.list()

nCores = 5
length = len(jtrainData)/nCores
chunks = list([jtrainData[x:x+length] for x in xrange(0, len(jtrainData), length)])

del jtrain, jtest, jtrainData, dtrain, dtest


def f(x, arg):
    print x, arg

#args, kw = (1,2,3), {'cat': 'dog'}

#print "# Normal call"
#f(range(0,5), 1)

print "############################## Multicall #################################"
pool = Pool(processes=5)
i=1
for chunk in chunks:
	pool.apply_async(getAllJsonTraining, (chunk, i))
	i+=1
#res = pool.apply_async(f, (range(0,5), 1))
#sol = [P.apply_async(f, (x,) + args, kw) for x in range(2)]
pool.close()
pool.join()
#print res.get()
#for s in sol: s.get()
