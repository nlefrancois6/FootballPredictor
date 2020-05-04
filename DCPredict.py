#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 09:35:00 2020

@author: noahlefrancois

Define the functions necessary for running the offensive play predictor and
assessing its accuracy.
"""
import numpy as np
import matplotlib.pyplot as plt

"""
Produce a pie chart showing the raw data for the chosen output variable
"""
def rawDataPie(Out, plotPie, testing_df):
    if Out == 'PLAY TYPE':
        if plotPie == True:
            rel_freq = testing_df['PLAY TYPE'].value_counts()
            f1=plt.figure()
            #need to edit the labels to match the categories in our data
            plt.pie(rel_freq, labels = ('Pass','Run'), autopct='%.2f%%')
            plt.title("Concordia 2019 play-type distribution")
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.show()
    elif Out == 'PLAY CATEGORY':
        if plotPie == True:
            playCat = testing_df['PLAY CATEGORY'].unique().tolist()
            playCatmapping = dict( zip(playCat,range(len(playCat))) )
            rel_freq = testing_df['PLAY CATEGORY'].value_counts()
            f1=plt.figure()
            #need to edit the labels to match the categories in our data
            #Remove 'SPECIAL' to run with CONUv1.csv
            plt.pie(rel_freq, labels = ('RPO','DROPBACK','RUN','PA POCKET','QUICK','SCREEN/DRAW','SPECIAL'), autopct='%.2f%%')
            plt.pie(rel_freq, autopct='%.2f%%')
            plt.title("Concordia 2019 play-type distribution")
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.show()
    else:
        print(Out + ' is not a supported output')

"""
Produce a bar graph for each model showing the feature importance weightings
"""
def featureImportancePlot(plotImportance, gbc, rfc, features):
    if plotImportance == True:
        #Determine how strongly each feature affects the outcome
        feature_importance = gbc.feature_importances_.tolist()
        f2=plt.figure()
        plt.bar(features,feature_importance)
        plt.title("gradient boosting classifier: feature importance")
        plt.xticks(rotation='vertical')
        plt.show()
    
        feature_importance = rfc.feature_importances_.tolist()
        f3=plt.figure()
        plt.bar(features,feature_importance)
        plt.title("random forest classifier: feature importance")
        plt.xticks(rotation='vertical')
        plt.show()
    
"""
For a given input play, check whether one of the n most likely labels is correct.
Return a boolean answer
"""
def orderedPredictionAccuracy(next_outcomes_prob, label_map, next_testing_label, n):
    pred_vector = []
    for i in range(0,n):
        pred_vector.append(label_map[int(next_outcomes_prob[i][1])])
    prediction_score = sum(np.isin(pred_vector, next_testing_label))
    
    return prediction_score

"""
For each play in the testing set, get the probabilities of each outcome and 
their labels in the mapping, select the n largest probabilities, and check if one of them
is the correct label. Return the average value of the boolean array prediction_scores
"""
def improved_Accuracy(pred_probs, label_map, testing_label, n):
    prediction_scores = np.empty(len(testing_label)) 
    for play in range(0, len(testing_label)):
        pred_probs_next = pred_probs[play]
        label_map_indices = np.linspace(0,len(pred_probs_next)-1,num=len(pred_probs_next))
        next_outcomes_prob = sorted(zip(pred_probs_next, label_map_indices), reverse=True)[:n]
        prediction_scores[play] = orderedPredictionAccuracy(next_outcomes_prob, label_map, testing_label.iloc[play], n)

    improved_accuracy = np.mean(prediction_scores)
    
    return improved_accuracy

