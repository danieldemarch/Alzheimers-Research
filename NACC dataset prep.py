# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:05:08 2019

@author: Montague
"""

import pandas as pd
import numpy as np

data = pd.read_csv("/Users/danieldemarchi/Desktop/Kos project/investigator_nacc45.csv")

alzheimers = data[data.NACCALZD != 0]
alzheimers3 = data[data.NACCALZD != 0]
print(len(alzheimers))
#dataset with everything
alzheimers = alzheimers[alzheimers.BILLS != -4]
alzheimers = alzheimers[alzheimers.TAXES != -4]
alzheimers = alzheimers[alzheimers.SHOPPING != -4]
alzheimers = alzheimers[alzheimers.GAMES != -4]
alzheimers = alzheimers[alzheimers.STOVE != -4]
alzheimers = alzheimers[alzheimers.MEALPREP != -4]
alzheimers = alzheimers[alzheimers.EVENTS != -4]
alzheimers = alzheimers[alzheimers.PAYATTN != -4]
alzheimers = alzheimers[alzheimers.REMDATES != -4]
alzheimers = alzheimers[alzheimers.TRAVEL != -4]
alzheimers = alzheimers[alzheimers.MEMORY != -4]
alzheimers = alzheimers[alzheimers.ORIENT != -4]
alzheimers = alzheimers[alzheimers.JUDGMENT != -4]
alzheimers = alzheimers[alzheimers.COMMUN != -4]
alzheimers = alzheimers[alzheimers.HOMEHOBB != -4]
alzheimers = alzheimers[alzheimers.PERSCARE != -4]
alzheimers = alzheimers[alzheimers.CDRSUM != -4]
alzheimers = alzheimers[alzheimers.CDRGLOB != -4]
alzheimers = alzheimers[alzheimers.COMPORT != -4]
alzheimers = alzheimers[alzheimers.CDRLANG != -4]
alzheimers = alzheimers[alzheimers.BILLS != 9]
alzheimers = alzheimers[alzheimers.TAXES != 9]
alzheimers = alzheimers[alzheimers.SHOPPING != 9]
alzheimers = alzheimers[alzheimers.GAMES != 9]
alzheimers = alzheimers[alzheimers.STOVE != 9]
alzheimers = alzheimers[alzheimers.MEALPREP != 9]
alzheimers = alzheimers[alzheimers.EVENTS != 9]
alzheimers = alzheimers[alzheimers.PAYATTN != 9]
alzheimers = alzheimers[alzheimers.REMDATES != 9]
alzheimers = alzheimers[alzheimers.TRAVEL != 9]
alzheimers = alzheimers[alzheimers.MEMORY != 9]
alzheimers = alzheimers[alzheimers.ORIENT != 9]
alzheimers = alzheimers[alzheimers.JUDGMENT != 9]
alzheimers = alzheimers[alzheimers.COMMUN != 9]
alzheimers = alzheimers[alzheimers.HOMEHOBB != 9]
alzheimers = alzheimers[alzheimers.PERSCARE != 9]
alzheimers = alzheimers[alzheimers.CDRSUM != 9]
alzheimers = alzheimers[alzheimers.CDRGLOB != 9]
alzheimers = alzheimers[alzheimers.COMPORT != 9]
alzheimers = alzheimers[alzheimers.CDRLANG != 9]

print(len(alzheimers))

id = alzheimers[['NACCID']]
counts = id['NACCID'].value_counts()
dropp = alzheimers.copy(deep = True)
edited = pd.DataFrame

dropees = counts.loc[counts.values == 1]
dropees = np.asarray(dropees.index)
print(dropees)
print(len(dropees))

i=0
rows_list = []

for index, row in dropp.iterrows():
    naccid = row['NACCID']
    if dropees.__contains__(naccid):
        i+=1
    else:
        dict1 = {}
        dict1.update(row)
        rows_list.append(dict1)

edited = pd.DataFrame(rows_list)               
edited.to_csv('/Users/danieldemarchi/Desktop/Kos project/Alz1')


