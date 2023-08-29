import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import datetime
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit
from pickle import dump, load

class ProcessedBestModel:
    def __init__(self):
        self.cols = [
            'INTERVAL', 'DISTRICT', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'PEDESTRIAN', 'CYCLIST', 
            'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH',
            'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY'
        ]
        self.mdl = []
        self.mdl.append(load(open('best_model_LogisticRegression.pkl', 'rb')))
        self.mdl.append(load(open('best_model_DecisionTree.pkl', 'rb')))
        self.mdl.append(load(open('best_model_RandomForest.pkl', 'rb')))
        self.mdl.append(load(open('best_model_NeuralNetwork.pkl', 'rb')))
        self.result = load(open('score_result.pkl', 'rb'))

        # self.lr = load(open('best_model_LogisticRegression.pkl', 'rb'))
        # self.dt = load(open('best_model_DecisionTree.pkl', 'rb'))
        # self.rf = load(open('best_model_RandomForest.pkl', 'rb'))
        # self.nn = load(open('best_model_NeuralNetwork.pkl', 'rb'))
        self.transformer = load(open('transformer.pkl', 'rb'))
    
    def predict(self, data):
        modelidx = int(data.pop(0))
        print(f'predict before: {modelidx}')

        input = [data]
        df = pd.DataFrame(input, columns=self.cols)
        data_transformed = self.transformer.transform(df)
        data_df = pd.DataFrame(data_transformed.toarray())
        # print('@@@@@@@@@@@@@@@@@@@@@@@@')
        # print(type(modelidx))
        # print(data)
        # print(data_df)
        # print('@@@@@@@@@@@@@@@@@@@@@@@@')
        pred = self.mdl[modelidx].predict(data_df)
        # if modelidx == 0:
        #     pred = self.lr.predict(data_df)
        #     print('HERE ~~~~~~~~~~~~~~~ ')
        # elif modelidx == 1:
        #     pred = self.dt.predict(data_df)
        # elif modelidx == 2:
        #     pred = self.rf.predict(data_df)
        # elif modelidx == 3:
        #     pred = self.nn.predict(data_df)
        # else:
        #     pred = 0

        print(f'predict end! {pred}')
        return str(int(pred[0]))
    
    def getresult(self, idx):
        return self.result[idx]

obj = ProcessedBestModel()
print(type(obj.getresult(0)["confusion_matrix"]))
print((obj.getresult(0)["confusion_matrix"]).replace("\n","<br />"))