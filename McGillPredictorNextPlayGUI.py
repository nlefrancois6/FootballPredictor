# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 22:27:23 2020
@author: Noah LeFrancois
@email: noah.lefrancois@mail.mcgill.ca
Will update changes made to McGillPredictor_GoodData here once validated, want to keep this file 
clean since it will be the end-product
Using data for each play in Con U's 2019 season (obtained from Hudl), we want to predict 
their play selection (run/pass, play type, zones targeted) on the next play given input info 
such as clock, field position, personnel, down&distance, defensive formation, etc. 
I'd like to use the first 7 games of their season to train the model, and test its predictions in
the final game of their season. Eventually, I'd like to update our model with each new play. 
By setting predNextPlay = True, we can take user input for the features of the upcoming play 
and predict the 3 most likely outcomes.
"""

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import ensemble
import pandas as pd
from sklearn import preprocessing
import numpy as np
import DCPredict as DC
import PySimpleGUI as sg

plotPie = False
plotImportance = False
predNextPlay = True
#Allowed Outputs: 'PLAY CATEGORY','PLAY TYPE'
Out = 'PLAY CATEGORY'
#Load the play data for the desired columns into a dataframe
#Currently the data is ordered by field zone so when i split into testing&training sets it's not
#randomly sampled. Need to either shuffle the csv entries or randomly sample from the df
df = pd.read_csv("CONUv3.csv")

#Get the variables we care about from the dataframe
df = df[['QTR','SCORE DIFF. (O)','SITUATION (O)','DRIVE #','DRIVE PLAY #','1ST DN #','D&D','Field Zone','HASH','OFF TEAM','PERS','OFF FORM','BACKF SET','PLAY CATEGORY','PLAY TYPE','DEF TEAM','DEF PERSONNEL', 'DEF FRONT']]

#Handle empty entries
#df = df.replace(np.nan, 'REG', regex=True)
df['SITUATION (O)'].fillna('REG', inplace=True)

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

#DEFFRONT = df['DEF FRONT'].unique().tolist()
#DEFFRONTmapping = dict( zip(DEFFRONT,range(len(DEFFRONT))) )
#df.replace({'DEF FRONT': DEFFRONTmapping},inplace=True)

#I'd like to compute the yards gained, play type, and result from the previous play and add them as an input for the current play
#df['prevYards'] = prevYards
#df['prevPlayType'] = prevPlayType
#df['prevPlayResult'] = prevPlayResult

#Separate into training data set (Con U 2019 Games 1-7) and testing data set (Con U 2019 Game 8)
#Random state is a seeding number
training_df = df.sample(frac=0.9, random_state=1)
indlist=list(training_df.index.values)

testing_df = df.copy().drop(index=indlist)


#Find the relative frequency of each outcome as a baseline to compare our play type prediction to

if Out == 'PLAY TYPE':
    rel_freq = testing_df['PLAY TYPE'].value_counts()
    if plotPie == True:
        f1=plt.figure()
        #need to edit the labels to match the categories in our data
        plt.pie(rel_freq, labels = ('Pass','Run'), autopct='%.2f%%')
        plt.title("Concordia 2019 play-type distribution")
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.show()
elif Out == 'PLAY CATEGORY':
    rel_freq = testing_df['PLAY CATEGORY'].value_counts()
    if plotPie == True:
        f1=plt.figure()
        #need to edit the labels to match the categories in our data
        plt.pie(rel_freq, labels = ('RPO','DROPBACK','RUN','PA POCKET','QUICK','SCREEN/DRAW'), autopct='%.2f%%')
        plt.title("Concordia 2019 play-type distribution")
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.show()
else:
    print(Out + ' is not a supported output')
    #sys.exit(['end'])

features = ['QTR','SCORE DIFF. (O)','SITUATION (O)','DRIVE #','DRIVE PLAY #','1ST DN #','D&D','Field Zone','PERS','DEF TEAM']

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
gbr = ensemble.GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.02, max_depth=1)

gbr.fit(training_features, training_label)

#Predict the outcome & probabilities from our test set
prediction = gbr.predict(testing_features)
pred_probs = gbr.predict_proba(testing_features)

#Get the label mappings for the prediction probabilities
le = preprocessing.LabelEncoder()
le.fit(training_label)
label_map = le.classes_

#Improved Accuracy Score for n top predictions
n=3

#Evaluate prediction accuracy
improved_accuracyGB = DC.improved_Accuracy(pred_probs, label_map, testing_label, n)
accuracyGB = accuracy_score(testing_label, prediction)
print("GBC Performance:")
print("Accuracy: "+"{:.2%}".format(accuracyGB)+", Improved Accuracy: "+"{:.2%}".format(improved_accuracyGB))


#Determine how strongly each feature affects the outcome 
feature_importance = gbr.feature_importances_.tolist()

if plotImportance == True:
    f2=plt.figure()
    plt.bar(features,feature_importance)
    plt.title("gradient boosting classifier: feature importance")
    plt.xticks(rotation='vertical')
    plt.show()

if predNextPlay == True:
    sg.theme('DarkBrown4')
    layout = [  [sg.Text('Enter Next Play Information')],
            [sg.Text('Quarter'), sg.Combo(['1', '2', '3', '4'])],
            [sg.InputText('Score Differential')],
            [sg.Text('Situation'), sg.Combo(['REG', 'OPENERS 1ST', 'OPENERS 2ND', '2 MIN'])],
            [sg.InputText('Drive Number')],
            [sg.InputText('Drive Play Number')],
            [sg.InputText('1st Downs in Drive')],
            [sg.Text('Down&Distance'), sg.Combo(['0&10','1&10', '1&11+', '1&9-', '2&2-', '2&3-6', '2&7+','3&2-','3&3-6','3&7+'])],
            [sg.Text('Field Position'), sg.Combo(['Backed Up (-1 to -19)', 'Coming Out (-20 to -39)','Open Field (-40 to 40)', 'Field Goal Fringe (39 to 21)', 'Red Zone (20 to 11)', 'Hot Zone (10 to 5)', 'Goal Line (4 to 1)'])],
            [sg.Text('Offensive Personnel'), sg.Combo(['15', '24 BIG', '24 SPEED', '33 JUMBO', '42'])],
            [sg.Text('Defensive Team'), sg.Combo(['UDM', 'SHERB', 'MCGILL', 'LVL'])],
            [sg.Button('Predict Next Play')] ]
    # Create the Window
    window = sg.Window('Window Title', layout)
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event in (None, 'Cancel'):	# if user closes window or clicks cancel
            break
        #print('You entered ', values)
    
        qtr = values[0]
        score = values[1]
        situation = values[2]
        driveNum = values[3]
        drivePlayNum = values[4]
        firstDownNum = values[5]
        dd = values[6]
        fieldZone = values[7]
        pers = values[8]
        defTeam = values[9]
        #defTeam = 'UDM'
    
        if event=='Predict Next Play':
            nextPlayFeatures = [qtr, score, situation, driveNum, drivePlayNum, firstDownNum, dd, fieldZone, pers, defTeam]
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
            dfNext.replace({'BACKF SET': BACKFSETmapping},inplace=True)
            dfNext.replace({'DEF PERSONNEL': DEFPERSONNELmapping},inplace=True)

            #Output the prediction
            pred_probs = gbr.predict_proba(dfNext)
    
            #Get the n most likely outcomes
            n=3
            pred_probs_next = pred_probs[0]
            label_map_indices = np.linspace(0,len(pred_probs_next)-1,num=len(pred_probs_next))
            next_outcomes_prob = sorted(zip(pred_probs_next, label_map_indices), reverse=True)[:n]

            print("Most Likely Outcomes: "+label_map[int(next_outcomes_prob[0][1])]+" "+"{:.2%}".format(next_outcomes_prob[0][0])+", "+label_map[int(next_outcomes_prob[1][1])]+" "+"{:.2%}".format(next_outcomes_prob[1][0])+", "+label_map[int(next_outcomes_prob[2][1])]+" "+"{:.2%}".format(next_outcomes_prob[2][0]))
            
    
    window.close()

