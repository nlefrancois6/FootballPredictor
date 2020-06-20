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

from sklearn.metrics import accuracy_score
from sklearn import ensemble
from sklearn.svm import SVC
import pandas as pd
from sklearn import preprocessing
import DCPredict as DC
import numpy as np
import time



# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


plotPie = False
plotImportance = False
#Allowed Outputs: 'PLAY CATEGORY','PLAY TYPE', 'ZONE THROWN'
#Need to add zone thrown to df list of variables in order to use it
Out = 'PLAY CATEGORY'
#Load the play data for the desired columns into a dataframe
#Currently the data is ordered by field zone so when i split into testing&training sets it's not
#randomly sampled. Need to either shuffle the csv entries or randomly sample from the df
df = pd.read_csv("CONUv4.csv")

#Get the variables we care about from the dataframe
df = df[['QTR','SCORE DIFF. (O)','SITUATION (O)','DRIVE #','DRIVE PLAY #','1ST DN #','D&D','Field Zone','HASH','OFF TEAM','PERS','PLAY CATEGORY','PLAY TYPE','DEF TEAM']]

#Handle empty entries
#df = df.replace(np.nan, 'REG', regex=True)
df['SITUATION (O)'].fillna('REG', inplace=True)
df = df.dropna()

#Specify which playlist sizes to test and make a copy of the original data 
#from which to draw the test plays
numPlays = [100, 200, 300, 400]
dfFull = df.copy()

#Give a list of random number generator seeds to test
seeds = [0,1]

#Initialize the arrays used to store the accuracy and time results
GBacc = np.zeros([len(seeds),2,len(numPlays)])
RFacc = np.zeros([len(seeds),2,len(numPlays)])
ETacc = np.zeros([len(seeds),2,len(numPlays)])
VCacc = np.zeros([len(seeds),2,len(numPlays)])

trainingTime = np.zeros([len(seeds),1,len(numPlays)])

#Keep track of which seed will be used
seedIndex = 0

#Train and test the model for each playlist size
for s in seeds:
    #Keep track of which index will be used to store the accuracy
    sizeIndex = 0
    
    for i in numPlays:
        #In order to change the number of plays for performance testing, take a subset of df
        df = dfFull.loc[1:i].copy()
    
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
        training_df = df.sample(frac=0.8, random_state=seeds[s])
        indlist=list(training_df.index.values)

        testing_df = df.copy().drop(index=indlist)

        #Shouldn't need to filter only run, pass like w/NFL data since we can select only offensive plays (no K or D). Need to make sure we're either excluding or handling dead plays though.


        #Find the relative frequency of labels as a baseline to compare our play type prediction to
        DC.rawDataPie(Out, plotPie, testing_df)


        #Define the features (input) and label (prediction output) for training set
        features = ['QTR','SCORE DIFF. (O)','SITUATION (O)','DRIVE #','DRIVE PLAY #', '1ST DN #','D&D','Field Zone','PERS','DEF TEAM']  
        training_features = training_df[features]
        #'QTR','SCORE DIFF. (O)','SITUATION (O)','DRIVE #','DRIVE PLAY #','1ST DN #','D&D','Field Zone','HASH','OFF TEAM','PERS','OFF FORM','BACKF SET','DEF TEAM','DEF PERSONNEL'

        training_label = training_df[Out]


        #Define features and label for testing set
        testing_features = testing_df[features]

        testing_label = testing_df[Out]


        #Start a timer to measure the model training time
        t = time.time()
    
        #Train a Gradient Boosting Machine on the data
        #Using 500 for category, 200 for type roughly maximizes accuracy so far
        #Default max_depth is 3, which works well for type, but a value of 1 gives 
        #better results for category
        gbc = ensemble.GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.02, max_depth=1)
        gbc.fit(training_features, training_label)

        #Train a Random Forest Machine on the data
        #max_depth=5 works well for type, value of 2 works well for category
        rfc = ensemble.RandomForestClassifier(n_estimators = 10, max_depth=2, random_state=120)
        rfc.fit(training_features, training_label)
        """
        #Can get really good (~77.5%) ImpAcc with BC, but bad (~37-40%) Acc 
        bcc = ensemble.BaggingClassifier(base_estimator=SVC(), n_estimators=500, random_state=0, max_samples=50, max_features = 10, bootstrap_features = True)
        bcc.fit(training_features, training_label)
        """
        #Train Extra Trees classifier on the data
        etc = ensemble.ExtraTreesClassifier(n_estimators=500, max_depth=5, random_state=0)
        etc.fit(training_features, training_label)

        #Soft Voting Predictor to combine GB and RF
        vc = ensemble.VotingClassifier(estimators=[('GB', gbc), ('RF', rfc), ('ET', etc)], voting='soft', weights=[8, 1, 4])
        vc.fit(training_features, training_label)


        #Predict the outcome from our test set and evaluate the prediction accuracy for each model
        predGB = gbc.predict(testing_features)
        pred_probsGB = gbc.predict_proba(testing_features)

        predRF = rfc.predict(testing_features)
        pred_probsRF = rfc.predict_proba(testing_features)

        #predBC = bcc.predict(testing_features)
        #pred_probsBC = bcc.predict_proba(testing_features)

        predET = etc.predict(testing_features)
        pred_probsET = etc.predict_proba(testing_features)

        predVC = vc.predict(testing_features)
        pred_probsVC = vc.predict_proba(testing_features)

        #Get the label mappings for the prediction probabilities
        le = preprocessing.LabelEncoder()
        le.fit(training_label)
        label_map = le.classes_

        #Improved Accuracy Score for n top predictions
        n=3

        #Accuracy for GBC
        improved_accuracyGB = DC.improved_Accuracy(pred_probsGB, label_map, testing_label, n)
        accuracyGB = accuracy_score(testing_label, predGB)
        GBacc[seedIndex,0,sizeIndex] = accuracyGB
        GBacc[seedIndex,1,sizeIndex] = improved_accuracyGB
     
        #Accuracy for RF
        improved_accuracyRF = DC.improved_Accuracy(pred_probsRF, label_map, testing_label, n)
        accuracyRF = accuracy_score(testing_label, predRF)
        RFacc[seedIndex,0,sizeIndex] = accuracyRF
        RFacc[seedIndex,1,sizeIndex] = improved_accuracyRF
    
        #Accuracy for ET
        improved_accuracyET = DC.improved_Accuracy(pred_probsET, label_map, testing_label, n)
        accuracyET = accuracy_score(testing_label, predET)
        ETacc[seedIndex,0,sizeIndex] = accuracyET
        ETacc[seedIndex,1,sizeIndex] = improved_accuracyET
    
        #Accuracy for VC
        improved_accuracyVC = DC.improved_Accuracy(pred_probsVC, label_map, testing_label, n)
        accuracyVC = accuracy_score(testing_label, predVC)
        VCacc[seedIndex,0,sizeIndex] = accuracyVC
        VCacc[seedIndex,1,sizeIndex] = improved_accuracyVC
    
        #Measure the model training time
        elapsed = time.time() - t
        trainingTime[seedIndex,0,sizeIndex] = elapsed
    
        #Plot feature importance for both models
        DC.featureImportancePlot(plotImportance, gbc, features)
    
        sizeIndex += 1
    
    seedIndex +=1

