#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 22:27:23 2020

@author: Noah LeFrancois
@email: noah.lefrancois@mail.mcgill.ca

Using data for each play in a dataset of past games, we want to predict 
the offensive play selection (run/pass, play type, zones targeted) on the next play given input info 
such as score, field position, personnel, down&distance, defensive team, etc. 

New data can be labelled and saved after each play, and this data can be added to re-train the model 
throughout a game in real-time.

By setting predNextPlay = True, we can take user input for the features of the upcoming play 
and predict the 3 most likely outcomes.

"""

from sklearn.metrics import accuracy_score
from sklearn import ensemble
from sklearn.neighbors import _typedefs
from sklearn.neighbors import _quad_tree
from sklearn.utils import _cython_blas
from sklearn.tree import _utils
#, sparsetools, lgamma 
#from sklearn.svm import SVC
import pandas as pd
from sklearn import preprocessing
import numpy as np
import DCPredict as DC
import PySimpleGUI as sg


# import warnings filter
#from warnings import simplefilter
# ignore all future warnings
#simplefilter(action='ignore', category=FutureWarning)

#predNextPlay must be True to run the GUI and in-game predictor
plotPie = False
plotImportance = False
predNextPlay = True
#Allowed Outputs: 'PLAY CATEGORY','PLAY TYPE'
Out = 'PLAY CATEGORY'
#Load the play data for the desired columns into a dataframe
#Currently the data is ordered by field zone so when i split into testing&training sets it's not
#randomly sampled. Need to either shuffle the csv entries or randomly sample from the df
df = pd.read_csv('CONUv3.csv')

#Get the variables we care about from the dataframe
df = df[['QTR','SCORE DIFF. (O)','SITUATION (O)','DRIVE #','DRIVE PLAY #','1ST DN #','D&D','Field Zone','HASH','OFF TEAM','PERS','OFF FORM','BACKF SET','PLAY CATEGORY','PLAY TYPE','DEF TEAM','DEF PERSONNEL', 'DEF FRONT', 'RESULT']]

#Handle empty entries
df['SITUATION (O)'].fillna('REG', inplace=True)
df['RESULT'].fillna('Rush', inplace=True)
df = df.dropna()

#Relabel the feature strings to number them so the GBR can read them
formations = df['OFF FORM'].unique().tolist()
formationmapping = dict( zip(formations,range(len(formations))) )
df.replace({'OFF FORM': formationmapping},inplace=True)

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

BACKFSET = df['BACKF SET'].unique().tolist()
BACKFSETmapping = dict( zip(BACKFSET,range(len(BACKFSET))) )
df.replace({'BACKF SET': BACKFSETmapping},inplace=True)

DEFPERSONNEL = df['DEF PERSONNEL'].unique().tolist()
DEFPERSONNELmapping = dict( zip(DEFPERSONNEL,range(len(DEFPERSONNEL))) )
df.replace({'DEF PERSONNEL': DEFPERSONNELmapping},inplace=True)


#Get the list of possible outcomes
Outcomes = df[Out].unique().tolist()
OutcomeMapping = dict( zip(Outcomes,range(len(Outcomes))) )


#Separate into training data set (Con U 2019 Games 1-7) and testing data set (Con U 2019 Game 8)
#Random state is a seeding number
training_df = df.sample(frac=0.8, random_state=1)
indlist=list(training_df.index.values)

testing_df = df.copy().drop(index=indlist)


#Find the relative frequency of labels as a baseline to compare our play type prediction to
DC.rawDataPie(Out, plotPie, testing_df)

features = ['QTR','SCORE DIFF. (O)','SITUATION (O)','DRIVE #','DRIVE PLAY #','1ST DN #','D&D','Field Zone','PERS','OFF TEAM', 'DEF TEAM']
columnLabels = features.copy()
columnLabels.append(Out)
#Define the features (input) and label (prediction output) for training set
training_features = training_df[features]
#'QTR','SCORE DIFF. (O)','SITUATION (O)','DRIVE #','DRIVE PLAY #','1ST DN #','D&D','Field Zone','HASH','OFF TEAM','PERS','OFF FORM','BACKF SET','DEF TEAM','DEF PERSONNEL'


if Out == 'PLAY TYPE':
    training_label = training_df['PLAY TYPE']
elif Out == 'PLAY CATEGORY':
    training_label = training_df['PLAY CATEGORY']


#Define features and label for testing set
testing_features = testing_df[features]

if Out == 'PLAY TYPE':
    testing_label = testing_df['PLAY TYPE']
elif Out == 'PLAY CATEGORY':
    testing_label = testing_df['PLAY CATEGORY']



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

#Train Extra Trees classifier on the data
etc = ensemble.ExtraTreesClassifier(n_estimators=500, max_depth=5, random_state=0)
etc.fit(training_features, training_label)

#Soft Voting Predictor to combine GB and RF
#8, 1, 4
vc = ensemble.VotingClassifier(estimators=[('GB', gbc), ('RF', rfc), ('ET', etc)], voting='soft', weights=[8, 1, 4])
vc.fit(training_features, training_label)


#Predict the outcome from our test set and evaluate the prediction accuracy for each model
predGB = gbc.predict(testing_features)
pred_probsGB = gbc.predict_proba(testing_features)

predRF = rfc.predict(testing_features)
pred_probsRF = rfc.predict_proba(testing_features)

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

#Accuracy for RF
improved_accuracyRF = DC.improved_Accuracy(pred_probsRF, label_map, testing_label, n)
accuracyRF = accuracy_score(testing_label, predRF)

#Accuracy for ET
improved_accuracyET = DC.improved_Accuracy(pred_probsET, label_map, testing_label, n)
accuracyET = accuracy_score(testing_label, predET)

#Accuracy for VC
improved_accuracyVC = DC.improved_Accuracy(pred_probsVC, label_map, testing_label, n)
accuracyVC = accuracy_score(testing_label, predVC)


#Plot feature importance for both models
#DC.featureImportancePlot(plotImportance, gbc, features)

if predNextPlay == True:
    #LightBrown13, LightGrey5, LightBlue3, Topanga, LightGrey5, DarkBlack1
    sg.theme('DarkBlack1')
    #Will need to read the possible inputs out of df to avoid errors when we get a formation we haven't seen before
    layout = [  [sg.Text('Enter Next Play Information')],
            [sg.Text('Defensive Team'), sg.Combo(defenseTeams), sg.Text('Offensive Team'), sg.Combo(OFFTEAM)],  
            [sg.Text('Quarter'), sg.Combo(['1', '2', '3', '4'])],
            [sg.Text('Score Differential'), sg.Slider(range=(-45, 45), orientation='h', size=(25, 20), default_value=0, tick_interval=15)],
            [sg.Text('Situation'), sg.Combo(situation)],
            [sg.Text('Drive Number'), sg.Slider(range=(1, 20), orientation='h', size=(25, 20), default_value=1, tick_interval=9)],
            [sg.Text('1st Downs in Drive'), sg.Slider(range=(0, 10), orientation='h', size=(25, 20), default_value=0, tick_interval=2)],
            [sg.Text('Drive Play Number'), sg.Slider(range=(1, 20), orientation='h', size=(25, 20), default_value=1, tick_interval=9)],
            [sg.Text('Down&Distance'), sg.Combo(DD)],
            [sg.Text('Field Position'), sg.Combo(FieldZone)],
            [sg.Text('Offensive Personnel'), sg.Combo(PERS)],
            [sg.Button('Predict Next Play'), sg.Button('Check Accuracy')] ,
            [sg.Output(size=(75, 6), font=('Helvetica 10'))], 
            [sg.Text('Play Outcome'), sg.Combo(Outcomes), sg.Button('Save Outcome')],
            [sg.Button('Add Saved Plays To Model'), sg.Button('Download Updated Data')] ]
    # Create the Window
    window = sg.Window('LeFrancois DC Play Predictor', layout)
    
    numPlaysSaved = 0
    inputsToSave = False
    numPlaysAdded = 0
    dfNewData = pd.DataFrame([], columns=columnLabels)    

    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event in (None, 'Cancel'):	# if user closes window or clicks cancel
            break
    
        defTeam = values[0]
        offTeam = values[1]
        qtr = values[2]
        score = values[3]
        situation = values[4]
        driveNum = values[5]
        firstDownNum = values[6]
        drivePlayNum = values[7]
        dd = values[8]
        fieldZone = values[9]
        pers = values[10]
        
        outcome = values[11]
    
        if event=='Predict Next Play':
            nextPlayFeatures = [qtr, score, situation, driveNum, drivePlayNum, firstDownNum, dd, fieldZone, pers, offTeam, defTeam]
            if '' in nextPlayFeatures:
                print("Cannot make prediction because not all inputs are filled")
            else:
                inputsToSave = True
                dfNext = pd.DataFrame([nextPlayFeatures], columns=features)
    
                #Relabel the feature strings with a numerical mapping
                dfNext.replace({'Formation': formationmapping},inplace=True)
                dfNext.replace({'DEF TEAM': defenseTeamMap},inplace=True)
                dfNext.replace({'SITUATION (O)': situationmapping},inplace=True)
                dfNext.replace({'D&D': DDmapping},inplace=True)
                dfNext.replace({'Field Zone': FieldZonemapping},inplace=True)
                dfNext.replace({'HASH': HASHmapping},inplace=True)
                dfNext.replace({'OFF TEAM': OFFTEAMmapping},inplace=True)
                dfNext.replace({'PERS': PERSmapping},inplace=True)

                #Output the prediction
                pred_probs = vc.predict_proba(dfNext)
    
                #Get the n most likely outcomes
                n=3
                pred_probs_next = pred_probs[0]
                label_map_indices = np.linspace(0,len(pred_probs_next)-1,num=len(pred_probs_next))
                next_outcomes_prob = sorted(zip(pred_probs_next, label_map_indices), reverse=True)[:n]

                print("Most Likely Outcomes: "+label_map[int(next_outcomes_prob[0][1])]+" "+"{:.2%}".format(next_outcomes_prob[0][0])+", "+label_map[int(next_outcomes_prob[1][1])]+" "+"{:.2%}".format(next_outcomes_prob[1][0])+", "+label_map[int(next_outcomes_prob[2][1])]+" "+"{:.2%}".format(next_outcomes_prob[2][0]))
        elif event=='Check Accuracy':
            print("Accuracy of Top 1: "+"{:.2%}".format(accuracyVC)+", Accuracy of Top 3: "+"{:.2%}".format(improved_accuracyVC))
        elif event=='Save Outcome':
            if inputsToSave == False:
                print("Cannot make prediction because inputs have not been used to predict a play.")
            elif outcome=='':
                print("Cannot make prediction because an outcome has not been recorded.")
            else:
                #Add the new play features and label to the training set
                labelNext = pd.DataFrame({0:[outcome]})
                training_features = training_features.append(dfNext, ignore_index = True)
                training_label = training_label.append(labelNext, ignore_index = True)
                #Add new play features and label to dfNewData, which can be downloaded later
                nextPlayFeatures.append(outcome)
                dfNextData = pd.DataFrame([nextPlayFeatures], columns=columnLabels)
                dfNewData = pd.concat([dfNewData, dfNextData])
                
                numPlaysSaved = numPlaysSaved + 1
                print('Play Outcome Saved. ' + str(numPlaysSaved) + " play(s) waiting to be added to model.")
                
                inputsToSave = False
        elif event=='Add Saved Plays To Model':
            #print('Adding new plays to model ...')
            #Train the model with expanded training set
            gbc.fit(training_features, training_label.values.ravel())
            rfc.fit(training_features, training_label.values.ravel())
            etc.fit(training_features, training_label.values.ravel())
            
            vc = ensemble.VotingClassifier(estimators=[('GB', gbc), ('RF', rfc), ('ET', etc)], voting='soft', weights=[8, 1, 4])
            vc.fit(training_features, training_label.values.ravel())

            predGB = gbc.predict(testing_features)
            pred_probsGB = gbc.predict_proba(testing_features)
            predRF = rfc.predict(testing_features)
            pred_probsRF = rfc.predict_proba(testing_features)
            predET = etc.predict(testing_features)
            pred_probsET = etc.predict_proba(testing_features)
            predVC = vc.predict(testing_features)
            pred_probsVC = vc.predict_proba(testing_features)

            le = preprocessing.LabelEncoder()
            le.fit(training_label.values.ravel())
            label_map = le.classes_

            improved_accuracyGB = DC.improved_Accuracy(pred_probsGB, label_map, testing_label, n)
            accuracyGB = accuracy_score(testing_label, predGB)
            improved_accuracyRF = DC.improved_Accuracy(pred_probsRF, label_map, testing_label, n)
            accuracyRF = accuracy_score(testing_label, predRF)
            improved_accuracyET = DC.improved_Accuracy(pred_probsET, label_map, testing_label, n)
            accuracyET = accuracy_score(testing_label, predET)
            improved_accuracyVC = DC.improved_Accuracy(pred_probsVC, label_map, testing_label, n)
            accuracyVC = accuracy_score(testing_label, predVC)
            
            print(str(numPlaysSaved) + " play(s) have been added to the model.")
            numPlaysAdded = numPlaysAdded + numPlaysSaved
            numPlaysSaved = 0
        elif event=='Download Updated Data':
            #Check if there's new data (numPlaysAddd>0), else print an error message
            if numPlaysAdded>0:                
                #Save the data to a csv in the dist folder
                dfNewData.to_csv('newData.csv', index=False)
                print('Data saved to dist folder.')
            else:
                print("No new data has been added yet.")
    window.close()
    
    

