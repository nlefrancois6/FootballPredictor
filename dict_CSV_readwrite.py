#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:41:32 2020

@author: noahlefrancois

Write or load csv containing a set of GridSearchCV results. 
"""
import csv

#task = 'open'
#OR
task = 'write'

if task == 'write':
    #Change the filename to your file
    with open('RFC_gridsearch.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        #Change the first two letters to your model
        for key, value in rfResults.items():
            writer.writerow([key, value])
elif task == 'open':
    #Change the filename to your file
    with open('GBC_gridsearch.csv') as csv_file:
        reader = csv.reader(csv_file)
        #Change the first two letters to your file's model
        gbResultsTest = dict(reader)