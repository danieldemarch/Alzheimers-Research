"""
Created on Thu Aug 15 17:40:13 2019

@author: Montague
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


alzheimers1 = pd.read_csv("/Users/danieldemarchi/Desktop/Kos project/missforestimpute.csv")
alzheimers1.TOBAC100 = alzheimers1.TOBAC100.replace(2, 1)
alzheimers1.TOBAC100 = alzheimers1.TOBAC100.replace(3, 1)

alzheimers1.CVPACE = alzheimers1.CVPACE.replace(4, 2)
alzheimers1.CVPACE = alzheimers1.CVPACE.replace(3, 2)

alzheimers1.CBTIA = alzheimers1.CBTIA.replace(3, 2)

alzheimers1.DIABTYPE = alzheimers1.DIABTYPE.replace(5, 3)
alzheimers1.DIABTYPE = alzheimers1.DIABTYPE.replace(4, 3)
alzheimers1.DIABTYPE = alzheimers1.DIABTYPE.replace(7, 8)

alzheimers1.HYPERTEN = alzheimers1.HYPERTEN.replace(3, 2)
alzheimers1.HYPERTEN = alzheimers1.HYPERTEN.replace(4, 2)

alzheimers1.HYPERCHO = alzheimers1.HYPERCHO.replace(3, 2)

alzheimers1.B12DEF = alzheimers1.B12DEF.replace(3, 2)
alzheimers1.B12DEF = alzheimers1.B12DEF.replace(4, 2)

alzheimers1.DEP2YRS = alzheimers1.DEP2YRS.replace(2, 1)
alzheimers1.DEP2YRS = alzheimers1.DEP2YRS.replace(3, 1)

alzheimers1.DEPOTHR = alzheimers1.DEPOTHR.replace(2, 1)
alzheimers1.DEPOTHR = alzheimers1.DEPOTHR.replace(3, 1)

alzheimers1.VISION = alzheimers1.VISION.replace(2, 1)
alzheimers1.VISION = alzheimers1.VISION.replace(3, 1)
alzheimers1.VISION = alzheimers1.VISION.replace(4, 1)
alzheimers1.VISION = alzheimers1.VISION.replace(5, 8)

alzheimers1.HEARING = alzheimers1.HEARING.replace(2, 1)
alzheimers1.HEARING = alzheimers1.HEARING.replace(3, 1)
alzheimers1.HEARING = alzheimers1.HEARING.replace(4, 1)

alzheimers1.HAPPY = alzheimers1.HAPPY.replace(2, 1)

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
naccmmse = alzheimers1.NACCMMSE

ti = ti/np.linalg.norm(ti)
age = age/np.linalg.norm(age)
height = height/np.linalg.norm(height)
weight = weight/np.linalg.norm(weight)
bpsys = bpsys/np.linalg.norm(bpsys)
bpdias = bpdias/np.linalg.norm(bpdias)
hrate = hrate/np.linalg.norm(hrate)
smokyrs = smokyrs/np.linalg.norm(smokyrs)
naccmmse = naccmmse/np.linalg.norm(naccmmse)


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
                       hrate, smokyrs, naccmmse]

    

for i in range(len(listy)):
    listy[i] = listy[i].to_numpy()
    if(listy[i].ndim == 1):
        listy[i] = np.reshape(listy[i], (len(listy[i]), 1))
    print(listy[i].shape[1])
print("Beginning concatenation")

cogscore1 = listy[0]
for i in range(1, len(listy)):
    cogscore1 = np.concatenate((cogscore1, listy[i]), axis=1)
    print(cogscore1.shape)

#print(cogscore1.shape)

#cogscore1 = cogscore1.transpose()

#print(cogscore1.shape)

np.save("/Users/danieldemarchi/Desktop/Kos project/onehotmissimpute", cogscore1)


















