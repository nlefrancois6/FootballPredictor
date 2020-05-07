#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:01:15 2020

@author: noahlefrancois
"""

import PySimpleGUI as sg

sg.theme('DarkBrown4')	# Add a touch of color
#DarkRed1, LightBrown13, DarkBrown4
#sg.popup('Hello From PySimpleGUI!', 'This is the shortest GUI program ever!')

# All the stuff inside your window.
#layout = [  [sg.Text('Some text on Row 1')],
#            [sg.Text('Enter something on Row 2'), sg.InputText()],
#            [sg.Button('Ok'), sg.Button('Cancel')] ]
layout = [  [sg.Text('Enter Next Play Information')],
            [sg.Text('Quarter'), sg.Combo(['1', '2', '3', '4'])],
            [sg.Text('Down&Distance'), sg.Combo(['1st&10', '1st&11+', '1st&9-', '2nd&3-', '2nd&4-7', '2nd&8+'])],
            [sg.Text('Field Position'), sg.Combo(['Backed Up', 'Open Field', 'Field Goal Range', 'Red Zone', 'Goal Line'])],
            [sg.Button('Predict Next Play')] ]

#Next: complete the input buttons and add an output display of the prediction

# Create the Window
window = sg.Window('Window Title', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event in (None, 'Cancel'):	# if user closes window or clicks cancel
        break
    print('You entered ', values)
    
    qtr = values[0]
    dd = values[1]
    field = values[2]
    
    if event=='Predict Next Play':
        break
    
window.close()
