#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 15:54:05 2020

@author: noahlefrancois
"""

import numpy as np
import matplotlib.pyplot as plt

errorBars = False

#plt.plot(numPlays, trainingTime[0,0,:])
#plt.plot(numPlays, trainingTime[1,0,:])

#Initialize arrays for storing measured values and their standard deviation
timeMean = np.zeros([1,len(numPlays)])
timeStd = np.zeros([1,len(numPlays)])

GBaccMean = np.zeros([1,len(numPlays)])
GBaccStd = np.zeros([1,len(numPlays)])
GBaccIMean = np.zeros([1,len(numPlays)])
GBaccIStd = np.zeros([1,len(numPlays)])

RFaccMean = np.zeros([1,len(numPlays)])
RFaccStd = np.zeros([1,len(numPlays)])
RFaccIMean = np.zeros([1,len(numPlays)])
RFaccIStd = np.zeros([1,len(numPlays)])

ETaccMean = np.zeros([1,len(numPlays)])
ETaccStd = np.zeros([1,len(numPlays)])
ETaccIMean = np.zeros([1,len(numPlays)])
ETaccIStd = np.zeros([1,len(numPlays)])

VCaccMean = np.zeros([1,len(numPlays)])
VCaccStd = np.zeros([1,len(numPlays)])
VCaccIMean = np.zeros([1,len(numPlays)])
VCaccIStd = np.zeros([1,len(numPlays)])


#Rescale the accuracy values to percentage instead of normalized fraction
GBacc = GBacc*100
RFacc = RFacc*100
ETacc = ETacc*100
VCacc = VCacc*100

#Calculate the average measured values and their std
for i in range(len(numPlays)):
    #Time
    timeMean[0,i] = np.mean(trainingTime[:,0,i])
    timeStd[0,i] = np.std(trainingTime[:,0,i])
    
    #Accuracy
    GBaccMean[0,i] = np.mean(GBacc[:,0,i])
    GBaccStd[0,i] = np.std(GBacc[:,0,i])
    GBaccIMean[0,i] = np.mean(GBacc[:,1,i])
    GBaccIStd[0,i] = np.std(GBacc[:,1,i])
    
    RFaccMean[0,i] = np.mean(RFacc[:,0,i])
    RFaccStd[0,i] = np.std(RFacc[:,0,i])
    RFaccIMean[0,i] = np.mean(RFacc[:,1,i])
    RFaccIStd[0,i] = np.std(RFacc[:,1,i])
    
    ETaccMean[0,i] = np.mean(ETacc[:,0,i])
    ETaccStd[0,i] = np.std(ETacc[:,0,i])
    ETaccIMean[0,i] = np.mean(ETacc[:,1,i])
    ETaccIStd[0,i] = np.std(ETacc[:,1,i])
    
    VCaccMean[0,i] = np.mean(VCacc[:,0,i])
    VCaccStd[0,i] = np.std(VCacc[:,0,i])
    VCaccIMean[0,i] = np.mean(VCacc[:,1,i])
    VCaccIStd[0,i] = np.std(VCacc[:,1,i])
    

fig, (ax1, ax2, ax3) = plt.subplots(3,1)
   
if errorBars:
    ax1.errorbar(numPlays, timeMean[0,:], timeStd[0,:], fmt='o:')
    ax1.set(xlabel='Number of Plays', ylabel='Model Training Time (s)')
    ax1.set_title('Training Time')
   
    ax2.errorbar(numPlays, GBaccMean[0,:], GBaccStd[0,:], fmt='o:', label='GB')
    ax2.errorbar(numPlays, RFaccMean[0,:], RFaccStd[0,:], fmt='o:', label='RF')
    ax2.errorbar(numPlays, ETaccMean[0,:], ETaccStd[0,:], fmt='o:', label='ET')
    ax2.errorbar(numPlays, VCaccMean[0,:], VCaccStd[0,:], fmt='o:', label='VC')
    ax2.set(xlabel='Number of Plays', ylabel='Model Accuracy (%)')
    ax2.set_title('Model Accuracy for Top Choice')
    ax2.legend(loc='lower right')
  
    ax3.errorbar(numPlays, GBaccIMean[0,:], GBaccIStd[0,:], fmt='o:', label='GB')
    ax3.errorbar(numPlays, RFaccIMean[0,:], RFaccIStd[0,:], fmt='o:', label='RF')
    ax3.errorbar(numPlays, ETaccIMean[0,:], ETaccIStd[0,:], fmt='o:', label='ET') 
    ax3.errorbar(numPlays, VCaccIMean[0,:], VCaccIStd[0,:], fmt='o:', label='VC')
    ax3.set(xlabel='Number of Plays', ylabel='Model Accuracy (%)')
    ax3.set_title('Model Accuracy for Top 3 Choices')
    ax3.legend(loc='lower right')

else:
    ax1.plot(numPlays, timeMean[0,:], 'o:')
    ax1.set(xlabel='Number of Plays', ylabel='Model Training Time (s)')
    ax1.set_title('Training Time')
   
    ax2.plot(numPlays, GBaccMean[0,:], 'o:', label='GB')
    ax2.plot(numPlays, RFaccMean[0,:], 'o:', label='RF')
    ax2.plot(numPlays, ETaccMean[0,:], 'o:', label='ET')
    ax2.plot(numPlays, VCaccMean[0,:], 'o:', label='VC')
    ax2.set(xlabel='Number of Plays', ylabel='Model Accuracy (%)')
    ax2.set_title('Model Accuracy for Top Choice')
    ax2.legend(loc='lower right')
  
    ax3.plot(numPlays, GBaccIMean[0,:], 'o:', label='GB')
    ax3.plot(numPlays, RFaccIMean[0,:], 'o:', label='RF')
    ax3.plot(numPlays, ETaccIMean[0,:], 'o:', label='ET') 
    ax3.plot(numPlays, VCaccIMean[0,:], 'o:', label='VC')
    ax3.set(xlabel='Number of Plays', ylabel='Model Accuracy (%)')
    ax3.set_title('Model Accuracy for Top 3 Choices')
    ax3.legend(loc='lower right')