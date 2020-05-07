#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 22:27:23 2020

@author: Noah LeFrancois
@email: noah.lefrancois@mail.mcgill.ca

Using data for each play in Con U's 2019 season (obtained from Hudl), we want to predict 
their play selection (run/pass, play type, zones targeted) on the next play given input info 
such as clock, field position, personnel, down&distance, defensive formation, etc. 

Here we use the first 7 games of their season to trainthe model, and test its predictions in
the final game of their season.
"""

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import ensemble
import pandas as pd
import numpy as np

plotPie = False
#Load the play data for the desired columns into a dataframe
df = pd.read_csv("pbp-2019.csv")

#Get the variables we care about from the dataframe
'''
df = df[['GameId','GameDate','Quarter','Minute','Second','OffenseTeam','DefenseTeam','Down','ToGo','YardLine','Formation','Yards','PlayType','IsTouchdown','SeriesFirstDown']]
#Compute the time elapsed in the game based on quarter, minute, second data
gameTime = 15*60*df['Quarter'] + 60*df['Minute'] + df['Second']

df['gameTime'] = gameTime

If we don't have quarter,minute, second data, we could just normalize to get the average play length over (#plays)/(60 minutes) to get a rough estimate
'''

#PlayType is (run, dropback, play-action, etc), PlayZone is target location of pass or run (str Flats, wk off-tackle, etc)
df = df[['GameId','GameDate','Quarter','Score','Situation','OffenseTeam','DefenseTeam','Down','ToGo','YardLine','SeriesFirstDown','Personnel','Formation','Yards','PlayType','PlayZone']]

df = df.dropna()
#df = df.replace(np.nan, 'REG', regex=True)

#I'd like to compute the yards gained, play type, and result from the previous play and add them as an input for the current play
#df['prevYards'] = prevYards
#df['prevPlayType'] = prevPlayType
#df['prevPlayResult'] = prevPlayResult

#Separate into training data set (Con U 2019 Games 1-7) and testing data set (Con U 2019 Game 8)
training_df = df[(~df.GameId.str.contains('Game 8')) & (df.OffenseTeam == 'ConU') & (df.Down.isin(range(1,4)))]
testing_df = df[(df.GameDate.str.contains('Game 8')) & (df.OffenseTeam == 'ConU') & (df.Down.isin(range(1,4)))]
#Shouldn't need to filter only run, pass like w/NFL data since we can select only offensive plays (no K or D). Need to make sure we're either excluding or handling dead plays though.


#Find the relative frequency of runs and passes as a baseline to compare our play type prediction to
rel_freq = testing_df['PlayType'].value_counts()

if plotPie == True:
    f1=plt.figure()
    #need to edit the labels to match the categories in our data
    plt.pie(rel_freq, labels = ('dropback','quick','PA','run'), autopct='%.2f%%')
    plt.title("Concordia 2019 play-type distribution")
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.show()


#Define the features (input) and label (prediction output) for training set
#Formation could be an input or output
training_features = training_df[['Quarter','Situation','Score','DefenseTeam','Down','ToGo','YardLine','SeriesPlays','Personnel','Formation']]

#Could predict any of these outputs:
#'Formation','Yards','PlayType','PlayZone'
training_label = training_df['PlayType']

#Define features and label for testing set
#If we only want to predict the next play, we could define testing_features and 
#testing_label with a new dataframe containing only the play we want. This information
#could also be taken from user input
testing_features = testing_df[['Quarter','Situation','Score','DefenseTeam','Down','ToGo','YardLine','SeriesPlays','Personnel','Formation']]

testing_label = testing_df['PlayType']

#Relabel the formation strings to number them so the GBR can read them
formations = training_features['Formation'].unique().tolist()
mapping = dict( zip(formations,range(len(formations))) )
training_features.replace({'Formation': mapping},inplace=True)
testing_features.replace({'Formation': mapping},inplace=True)

defenseTeams = training_features['DefenseTeam'].unique().tolist()
defenseTeamMap = dict( zip(defenseTeams,range(len(defenseTeams))) )
training_features.replace({'DefenseTeam': defenseTeamMap},inplace=True)
testing_features.replace({'DefenseTeam': defenseTeamMap},inplace=True)

"""
#Relabel the formation strings to number them so the GBR can read them
formations = df['OFF FORM'].unique().tolist()
mapping = dict( zip(formations,range(len(formations))) )
training_features.replace({'OFF FORM': mapping},inplace=True)
testing_features.replace({'OFF FORM': mapping},inplace=True)

defenseTeams = training_features['DEF TEAM'].unique().tolist()
defenseTeamMap = dict( zip(defenseTeams,range(len(defenseTeams))) )
training_features.replace({'DEF TEAM': defenseTeamMap},inplace=True)
testing_features.replace({'DEF TEAM': defenseTeamMap},inplace=True)

DD = training_features['D&D'].unique().tolist()
DDmapping = dict( zip(DD,range(len(DD))) )
training_features.replace({'D&D': DDmapping},inplace=True)
testing_features.replace({'D&D': DDmapping},inplace=True)

FieldZone = training_features['Field Zone'].unique().tolist()
FieldZonemapping = dict( zip(FieldZone,range(len(FieldZone))) )
training_features.replace({'Field Zone': FieldZonemapping},inplace=True)
testing_features.replace({'Field Zone': FieldZonemapping},inplace=True)

HASH = training_features['HASH'].unique().tolist()
HASHmapping = dict( zip(HASH,range(len(HASH))) )
training_features.replace({'HASH': HASHmapping},inplace=True)
testing_features.replace({'HASH': HASHmapping},inplace=True)

OFFTEAM = training_features['OFF TEAM'].unique().tolist()
OFFTEAMmapping = dict( zip(OFFTEAM,range(len(OFFTEAM))) )
training_features.replace({'OFF TEAM': OFFTEAMmapping},inplace=True)
testing_features.replace({'OFF TEAM': OFFTEAMmapping},inplace=True)

PERS = training_features['PERS'].unique().tolist()
PERSmapping = dict( zip(PERS,range(len(PERS))) )
training_features.replace({'PERS': PERSmapping},inplace=True)
testing_features.replace({'PERS': PERSmapping},inplace=True)

BACKFSET = training_features['BACKF SET'].unique().tolist()
BACKFSETmapping = dict( zip(BACKFSET,range(len(BACKFSET))) )
training_features.replace({'BACKF SET': BACKFSETmapping},inplace=True)
testing_features.replace({'BACKF SET': BACKFSETmapping},inplace=True)

DEFPERSONNEL = training_features['DEF PERSONNEL'].unique().tolist()
DEFPERSONNELmapping = dict( zip(DEFPERSONNEL,range(len(DEFPERSONNEL))) )
training_features.replace({'DEF PERSONNEL': DEFPERSONNELmapping},inplace=True)
testing_features.replace({'DEF PERSONNEL': DEFPERSONNELmapping},inplace=True)
"""

#Will need to relabel any other strings used as input
#Doesn't look like I need to relabel strings used as output though


#Train a Gradient Boosting Machine on the data
gbr = ensemble.GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.02)

gbr.fit(training_features, training_label)

#Predict the run/pass percentage from our test set and evaluate the prediction accuracy
prediction = gbr.predict(testing_features)
#Get a prediction for a particular play:
#print(prediction[0])
"""
#Give a set of features for the next play and predict the outcome
nextPlayFeatures = [4,'2Min','-7','2','mid','43','6','15','32']
predNextPlay = gbr.predict(nextPlayFeatures)
print("Most likely next play:" + predNextPlay)
"""

"""
#Predict the outcome from our test set and evaluate the prediction accuracy
prediction = gbr.predict(testing_features)
pred_probs = gbr.predict_proba(testing_features)

#Get the label mappings for the prediction probabilities
le = preprocessing.LabelEncoder()
le.fit(training_label)
label_map = le.classes_

#Get the n most likely outcomes for the 1st play

pred_probs_next = pred_probs[0]
label_map_indices = np.linspace(0,len(pred_probs_next)-1,num=len(pred_probs_next))
next_outcomes_prob = sorted(zip(pred_probs_next, label_map_indices), reverse=True)[:n]

print("Most Likely Outcomes: "+label_map[int(next_outcomes_prob[0][1])]+" "+"{:.2%}".format(next_outcomes_prob[0][0])+", "+label_map[int(next_outcomes_prob[1][1])]+" "+"{:.2%}".format(next_outcomes_prob[1][0])+", "+label_map[int(next_outcomes_prob[2][1])]+" "+"{:.2%}".format(next_outcomes_prob[2][0]))
"""

accuracy = accuracy_score(testing_label, prediction)

print("Accuracy: "+"{:.2%}".format(accuracy))

#Determine how strongly each feature affects the outcome

features = ['Quarter','Situation','Score','DefenseTeam','Down','ToGo','YardLine','SeriesPlays','Personnel','Formation'] 

feature_importance = gbr.feature_importances_.tolist()


plt.bar(features,feature_importance)
plt.title("gradient boosting classifier: feature importance")
plt.show()


    nextPlayFeatures = [Quarter, Score, Situation, DriveNum, DrivePlayNum, firstDownNum, DD, FieldZone, Hash, OTeam, Pers, DTeam, DPers]
    #Relabel the feature strings with a numerical mapping
    nextPlayFeatures.replace({'Formation': formationmapping},inplace=True)
    nextPlayFeatures.replace({'DEF TEAM': defenseTeamMap},inplace=True)
    nextPlayFeatures.replace({'SITUATION (O)': situationmapping},inplace=True)
    nextPlayFeatures.replace({'D&D': DDmapping},inplace=True)
    nextPlayFeatures.replace({'Field Zone': FieldZonemapping},inplace=True)
    nextPlayFeatures.replace({'HASH': HASHmapping},inplace=True)
    nextPlayFeatures.replace({'OFF TEAM': OFFTEAMmapping},inplace=True)
    nextPlayFeatures.replace({'PERS': PERSmapping},inplace=True)
    nextPlayFeatures.replace({'BACKF SET': BACKFSETmapping},inplace=True)
    nextPlayFeatures.replace({'DEF PERSONNEL': DEFPERSONNELmapping},inplace=True)
    
    #indlistDF=list(df.index.values)
    #nextPlay_df = df.copy().drop(index=indlistDF)
    #nextPlay_df = df.insert(loc=0, column='QTR', value=1)

