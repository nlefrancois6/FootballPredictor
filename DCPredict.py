#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 09:35:00 2020

@author: noahlefrancois

Define the functions necessary for running the offensive play predictor and
assessing its accuracy.
"""
from numpy import empty, linspace, isin, mean
from matplotlib.pyplot import figure, pie, title, subplots_adjust, show, bar, xticks

"""
Produce a pie chart showing the raw data for the chosen output variable
"""
def rawDataPie(Out, plotPie, testing_df):
    if plotPie == True:
        playCat = testing_df[Out].unique().tolist()
        #playCatmapping = dict( zip(playCat,range(len(playCat))) )
        rel_freq = testing_df[Out].value_counts()
        labelList = []
        for label in range(0,len(playCat)):
            labelList.append(rel_freq.index[label])
        f1=figure()
        pie(rel_freq, labels = labelList, autopct='%.2f%%')
        title("Concordia 2019 play-type distribution")
        subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        show()

"""
Produce a bar graph for each model showing the feature importance weightings
"""
def featureImportancePlot(plotImportance, gbc, features):
    if plotImportance == True:
        #Determine how strongly each feature affects the outcome
        feature_importance = gbc.feature_importances_.tolist()
        f2=figure()
        bar(features,feature_importance)
        title("Gradient Boosting Classifier: Feature Importance")
        xticks(rotation='vertical')
        show()
    
"""
For a given input play, check whether one of the n most likely labels is correct.
Return a boolean answer
"""
def orderedPredictionAccuracy(next_outcomes_prob, label_map, next_testing_label, n):
    pred_vector = []
    for i in range(0,n):
        pred_vector.append(label_map[int(next_outcomes_prob[i][1])])
    prediction_score = sum(isin(pred_vector, next_testing_label))
    
    return prediction_score

"""
For each play in the testing set, get the probabilities of each outcome and 
their labels in the mapping, select the n largest probabilities, and check if one of them
is the correct label. Return the average value of the boolean array prediction_scores
"""
def improved_Accuracy(pred_probs, label_map, testing_label, n):
    prediction_scores = empty(len(testing_label)) 
    for play in range(0, len(testing_label)):
        pred_probs_next = pred_probs[play]
        label_map_indices = linspace(0,len(pred_probs_next)-1,num=len(pred_probs_next))
        next_outcomes_prob = sorted(zip(pred_probs_next, label_map_indices), reverse=True)[:n]
        prediction_scores[play] = orderedPredictionAccuracy(next_outcomes_prob, label_map, testing_label.iloc[play], n)

    improved_accuracy = mean(prediction_scores)
    
    return improved_accuracy