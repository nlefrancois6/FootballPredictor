#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 22:27:23 2020
@author: Noah LeFrancois
@email: noah.lefrancois@mail.mcgill.ca
Using data for each play in Con U's 2019 season (obtained from Hudl), we want to predict 
their play selection (run/pass, play type, zones targeted) on the next play given input info 
such as clock, field position, personnel, down&distance, score, etc. 
"""

from sklearn import ensemble 
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn import preprocessing
import DCPredict as DC

# import warnings filter (I don't think I need this anymore since it was for the BCC)
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)


plotPie = False
plotImportance = False
plotConfusion = True
#Allowed Outputs: 'PLAY CATEGORY','PLAY TYPE', 'ZONE THROWN'
#Need to add zone thrown to df list of variables in order to use it
Out = 'PLAY CATEGORY'
#Load the play data for the desired columns into a dataframe
#CONU_SHERB, CONUv5
df = pd.read_csv("CONU_SHERB.csv")

#Get the variables we care about from the dataframe
df = df[['QTR','SCORE DIFF. (O)','SITUATION (O)','DRIVE #','DRIVE PLAY #','1ST DN #','D&D','Field Zone','HASH','OFF TEAM','PERS','PLAY CATEGORY','PLAY TYPE','DEF TEAM']]

#Select specific subsections of the data set, ie just one specific offensive team  
#SHER, CONU       
#df.drop(df[df['OFF TEAM'] != 'CONU'].index, inplace=True)
         
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


#Define the features (input) and label (prediction output) for training set
features = ['QTR','SCORE DIFF. (O)','SITUATION (O)','DRIVE #','DRIVE PLAY #', '1ST DN #','D&D','Field Zone','PERS','DEF TEAM','OFF TEAM']  
training_features, testing_features, training_label, testing_label = train_test_split(df[features], df[Out], test_size=0.2, random_state=0, stratify=df[Out])
#'QTR','SCORE DIFF. (O)','SITUATION (O)','DRIVE #','DRIVE PLAY #','1ST DN #','D&D','Field Zone','HASH','OFF TEAM','PERS','OFF FORM','BACKF SET','DEF TEAM','DEF PERSONNEL'

#Find the relative frequency of labels as a baseline to compare our play type prediction to
DC.rawDataPie(Out, plotPie, testing_label)


#Train a Gradient Boosting Machine on the data
parG = {'n_estimators':[1,10,100,500,800], 'learning_rate':[0.01,0.03,0.05,0.08,0.1],'max_depth':[1,2,3,4,5,8]}
gbc = ensemble.GradientBoostingClassifier()
gbSearch = GridSearchCV(gbc, parG)
gbSearch.fit(training_features, training_label)

gbResults = gbSearch.cv_results_

#Train a Random Forest Machine on the data
parR = {'n_estimators':[5,15,30,45,60,75,90,115,140,170,200], 'max_depth':[2,4,6,8,10,12], 'random_state':[120]}
rfc = ensemble.RandomForestClassifier()
rfSearch = GridSearchCV(rfc, parR)
rfSearch.fit(training_features, training_label)

rfResults = rfSearch.cv_results_

#Train Extra Trees classifier on the data
parE = {'n_estimators':[10,100,200,500,800,1000,1200,1500],'max_depth':[2,4,6,8,10,12],'random_state':[0]}
etc = ensemble.ExtraTreesClassifier()
etSearch = GridSearchCV(etc, parE)
etSearch.fit(training_features, training_label)

etResults = etSearch.cv_results_

#Once the 3 classifiers have been optimized, I can also use a gridsearch for the VC weights.
