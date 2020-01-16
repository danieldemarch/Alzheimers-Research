"""
Created on Thu Aug 15 17:40:13 2019

@author: Montague
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


alzheimers1 = pd.read_csv("/Users/danieldemarchi/Desktop/Kos project/Alz1")

naccid = alzheimers1.NACCID
cdrglob = pd.get_dummies(alzheimers1.CDRGLOB)
age = alzheimers1.NACCAGE
ti = alzheimers1.NACCFDYS
race = pd.get_dummies(alzheimers1.RACE)
lang = pd.get_dummies(alzheimers1.PRIMLANG)
livs = pd.get_dummies(alzheimers1.NACCLIVS)
independ = pd.get_dummies(alzheimers1.INDEPEND)
residenc = pd.get_dummies(alzheimers1.RESIDENC)
hand = pd.get_dummies(alzheimers1.HANDED)
tobac30 = pd.get_dummies(alzheimers1.TOBAC30)
tobac100 = pd.get_dummies(alzheimers1.TOBAC100)
smokyrs = alzheimers1.SMOKYRS
cvhatt = pd.get_dummies(alzheimers1.CVHATT)
cvafib = pd.get_dummies(alzheimers1.CVAFIB)
cvbypass = pd.get_dummies(alzheimers1.CVBYPASS)
cvpace = pd.get_dummies(alzheimers1.CVPACE)
cvchf = pd.get_dummies(alzheimers1.CVCHF)
cvothr = pd.get_dummies(alzheimers1.CVOTHR)
cbstroke = pd.get_dummies(alzheimers1.CBSTROKE)
cbtia = pd.get_dummies(alzheimers1.CBTIA)
seizures = pd.get_dummies(alzheimers1.SEIZURES)
diabetes = pd.get_dummies(alzheimers1.DIABETES)
diabtype = pd.get_dummies(alzheimers1.DIABTYPE)
hyperten = pd.get_dummies(alzheimers1.HYPERTEN)
hypercho = pd.get_dummies(alzheimers1.HYPERCHO)
b12def = pd.get_dummies(alzheimers1.B12DEF)
thyroid = pd.get_dummies(alzheimers1.THYROID)
alcohol = pd.get_dummies(alzheimers1.ALCOHOL)
dep2yrs = pd.get_dummies(alzheimers1.DEP2YRS)
depothr = pd.get_dummies(alzheimers1.DEPOTHR)
psycdis = pd.get_dummies(alzheimers1.PSYCDIS)
height = alzheimers1.HEIGHT
weight = alzheimers1.WEIGHT
bpsys = alzheimers1.BPSYS
bpdias = alzheimers1.BPDIAS
hrate = alzheimers1.HRATE
vision = pd.get_dummies(alzheimers1.VISION)
hearing = pd.get_dummies(alzheimers1.HEARING)
memory = pd.get_dummies(alzheimers1.MEMORY)
orient = pd.get_dummies(alzheimers1.ORIENT)
judgment = pd.get_dummies(alzheimers1.JUDGMENT)
commun = pd.get_dummies(alzheimers1.COMMUN)
home = pd.get_dummies(alzheimers1.HOMEHOBB)
perscare = pd.get_dummies(alzheimers1.PERSCARE)
cdrsum = pd.get_dummies(alzheimers1.CDRSUM)
cdrlang = pd.get_dummies(alzheimers1.CDRLANG)
comport = pd.get_dummies(alzheimers1.COMPORT)
nogds = pd.get_dummies(alzheimers1.NOGDS)
satis = pd.get_dummies(alzheimers1.SATIS)
dropact = pd.get_dummies(alzheimers1.DROPACT)
empty = pd.get_dummies(alzheimers1.EMPTY)
bored = pd.get_dummies(alzheimers1.BORED)
spirits = pd.get_dummies(alzheimers1.SPIRITS)
afraid = pd.get_dummies(alzheimers1.AFRAID)
happy = pd.get_dummies(alzheimers1.HAPPY)
helpless = pd.get_dummies(alzheimers1.HELPLESS)
stayhome = pd.get_dummies(alzheimers1.STAYHOME)
memprob = pd.get_dummies(alzheimers1.MEMPROB)
wondrful = pd.get_dummies(alzheimers1.WONDRFUL)
wrthless = pd.get_dummies(alzheimers1.WRTHLESS)
energy = pd.get_dummies(alzheimers1.ENERGY)
hopeless = pd.get_dummies(alzheimers1.HOPELESS)
better = pd.get_dummies(alzheimers1.BETTER)
naccgds = pd.get_dummies(alzheimers1.NACCGDS)
bills = pd.get_dummies(alzheimers1.BILLS)
taxes = pd.get_dummies(alzheimers1.TAXES)
shopping = pd.get_dummies(alzheimers1.SHOPPING)
games = pd.get_dummies(alzheimers1.GAMES)
stove = pd.get_dummies(alzheimers1.STOVE)
mealprep = pd.get_dummies(alzheimers1.MEALPREP)
events = pd.get_dummies(alzheimers1.EVENTS)
payattn = pd.get_dummies(alzheimers1.PAYATTN)
remdates = pd.get_dummies(alzheimers1.REMDATES)
travel = pd.get_dummies(alzheimers1.TRAVEL)
naccmmse = pd.get_dummies(alzheimers1.NACCMMSE)

ti = ti/np.linalg.norm(ti)
age = age/np.linalg.norm(age)
height = height/np.linalg.norm(height)
weight = weight/np.linalg.norm(weight)
bpsys = bpsys/np.linalg.norm(bpsys)
bpdias = bpdias/np.linalg.norm(bpdias)
hrate = hrate/np.linalg.norm(hrate)
smokyrs = smokyrs/np.linalg.norm(smokyrs)


listy = [naccid, ti, age, race, lang, livs, 
                       independ, residenc, hand, tobac30,
                       tobac100, smokyrs, cvhatt, cvafib,
                       cvbypass, cvpace, cvchf, cvothr,
                       cbstroke, cbtia, seizures, diabetes,
                       diabtype, hyperten, hypercho, b12def,
                       thyroid, alcohol, dep2yrs, depothr,
                       psycdis, height, weight, bpsys, bpdias,
                       hrate, vision, hearing, nogds, satis,
                       dropact, empty, bored, spirits, afraid,
                       happy, helpless, stayhome, memprob,
                       wondrful, wrthless, energy, hopeless, better, naccgds, 
                       naccmmse, bills, taxes, shopping, games, stove, 
                       mealprep, events, payattn, remdates,
                       travel, memory, orient, judgment, commun, 
                       home, perscare, cdrsum, cdrlang, comport, 
                       cdrglob]

continvars = [ti, age, height, weight, bpsys, bpdias,
                       hrate, smokyrs]

ordvars = []

nomvars = []

    

for i in range(len(listy)):
    listy[i] = listy[i].to_numpy()
    if(listy[i].ndim == 1):
        listy[i] = np.reshape(listy[i], (len(listy[i]), 1))
    print(listy[i].shape)

cogscore1 = listy[0]
for i in range(1, len(listy)):
    cogscore1 = np.concatenate((cogscore1, listy[i]), axis=1)
    print(cogscore1.shape)

#print(cogscore1.shape)

#cogscore1 = cogscore1.transpose()

#print(cogscore1.shape)

np.save("/Users/danieldemarchi/Desktop/Kos project/onehotnoimpute", cogscore1)


















