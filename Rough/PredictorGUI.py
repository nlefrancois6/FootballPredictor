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
#            [sg.Button('Ok'), sg.Button('Cancel')],
#             [sg.Text('Situation'), sg.Combo(['', '', '', ''])] ]
layout = [  [sg.Text('Enter Next Play Information')],
            [sg.Text('Quarter'), sg.Combo(['1', '2', '3', '4'])],
            [sg.InputText('Score Differential')],
            [sg.Text('Situation'), sg.Combo(['Regular', 'Openers 1st Half', 'Openers 2nd Half', '2 Minute'])],
            [sg.InputText('Drive Number')],
            [sg.InputText('Drive Play Number')],
            [sg.InputText('1st Downs in Drive')],
            [sg.Text('Down&Distance'), sg.Combo(['0&10','1&10', '1&11+', '1&9-', '2&2-', '2&3-6', '2&7+','3&2-','3&3-6','3&7+'])],
            [sg.Text('Field Position'), sg.Combo(['Backed Up (-1 to -19)', 'Coming Out (-20 to -39)','Open Field (-40 to 40)', 'Field Goal Fringe (39 to 21)', 'Red Zone (20 to 11)', 'Hot Zone (10 to 5)', 'Goal Line (4 to 1)'])],
            [sg.Text('Offensive Personnel'), sg.Combo(['15', '24 BIG', '24 SPEED', '33 JUMBO', '42'])]
            [sg.Text('Defensive Team'), sg.Combo(['UDM', 'SHERB', 'MCGILL', 'LVL'])]
            [sg.Button('Predict Next Play')] ]

#Next steps: complete the input buttons and add an output display of the prediction

# Create the Window
window = sg.Window('Window Title', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event in (None, 'Cancel'):	# if user closes window or clicks cancel
        break
    print('You entered ', values)
    
    qtr = values[0]
    score = values[1]
    situation = values[2]
    driveNum = values[3]
    drivePlay = values[4]
    firstDowns = values[5]
    dd = values[6]
    fieldZone = values[7]
    pers = values[8]
    defTeam = values[9]
    
    if event=='Predict Next Play':
        
        break
    
window.close()
