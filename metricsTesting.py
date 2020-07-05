#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 00:21:40 2020

@author: noahlefrancois
"""

from sklearn import ensemble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import DCPredict as DC
from sklearn.metrics import confusion_matrix, jaccard_score, f1_score
import itertools

Out = 'PLAY CATEGORY'

df = pd.read_csv("CONUv4.csv")

#Get the variables we care about from the dataframe
df = df[['QTR','SCORE DIFF. (O)','SITUATION (O)','DRIVE #','DRIVE PLAY #','1ST DN #','D&D','Field Zone','HASH','OFF TEAM','PERS','PLAY CATEGORY','PLAY TYPE','DEF TEAM']]

#Handle empty entries
#df = df.replace(np.nan, 'REG', regex=True)
df['SITUATION (O)'].fillna('REG', inplace=True)
df = df.dropna()

#Relabel the feature strings to number them so the GBR can read them
situation = df['SITUATION (O)'].unique().tolist()
situationmapping = dict( zip(situation,range(len(situation))) )
df.replace({'SITUATION (O)': situationmapping},inplace=True)

defenseTeams = df['DEF TEAM'].unique().tolist()
defenseTeamMap = dict( zip(defenseTeams,range(len(defenseTeams))) )
df.replace({'DEF TEAM': defenseTeamMap},inplace=True)

DD = df['D&D'].unique().tolist()
DDmapping = dict( zip(DD,range(len(DD))) )
df.replace({'D&D': DDmapping},inplace=True)

FieldZone = df['Field Zone'].unique().tolist()
FieldZonemapping = dict( zip(FieldZone,range(len(FieldZone))) )
df.replace({'Field Zone': FieldZonemapping},inplace=True)

HASH = df['HASH'].unique().tolist()
HASHmapping = dict( zip(HASH,range(len(HASH))) )
df.replace({'HASH': HASHmapping},inplace=True)

OFFTEAM = df['OFF TEAM'].unique().tolist()
OFFTEAMmapping = dict( zip(OFFTEAM,range(len(OFFTEAM))) )
df.replace({'OFF TEAM': OFFTEAMmapping},inplace=True)

PERS = df['PERS'].unique().tolist()
PERSmapping = dict( zip(PERS,range(len(PERS))) )
df.replace({'PERS': PERSmapping},inplace=True)



#Separate into training data set and testing data set
training_df = df.sample(frac=0.8, random_state=0)
indlist=list(training_df.index.values)

testing_df = df.copy().drop(index=indlist)

#Define the features (input) and label (prediction output) for training set
features = ['QTR','SCORE DIFF. (O)','SITUATION (O)','DRIVE #','DRIVE PLAY #', '1ST DN #','D&D','Field Zone','PERS','DEF TEAM']  
training_features = training_df[features]
#'QTR','SCORE DIFF. (O)','SITUATION (O)','DRIVE #','DRIVE PLAY #','1ST DN #','D&D','Field Zone','HASH','OFF TEAM','PERS','OFF FORM','BACKF SET','DEF TEAM','DEF PERSONNEL'

training_label = training_df[Out]


#Define features and label for testing set
testing_features = testing_df[features]

testing_label = testing_df[Out]

#Get the label mappings for the prediction probabilities
le = preprocessing.LabelEncoder()
le.fit(training_label)
label_map = le.classes_

#Train a Gradient Boosting Machine on the data
gbc = ensemble.GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.02, max_depth=1)
gbc.fit(training_features, training_label)

predGB = gbc.predict(testing_features)
pred_probsGB = gbc.predict_proba(testing_features)


#Evaluate model performance

# Plot non-normalized confusion matrix
DC.confusionMatrix(True, testing_label, predGB, label_map)

#Scores
print("F1 Score: %.4f" % f1_score(testing_label, predGB, average='weighted'))
print("Jaccard Score: %.4f" % jaccard_score(testing_label, predGB, average='weighted'))