"""
Created on Thu Aug 15 14:15:27 2019
@author: Montague
"""
#FAQ = 7 vars apiece, 9 and -4 need to be imputed
#EDIT - FAQ, -4 IS EXCLUDED BUT YOU SHOULD EXCLUDE 9 TOO!
#everything except perscare = 5 vars apiece
#perscare = 4 vars

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

import pandas as pd
import numpy as np

from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten
from  keras.optimizers import RMSprop
import keras
import math

cogscores4 = np.load("/Users/Montague/Desktop/DStuff/Cog1.npy")

np.set_printoptions(suppress=True)

future_cog41 = []
future_cog42 = []
future_cog43 = []
future_cog44 = []
future_cog45 = []

ilist1 = []
ilist2 = []
ilist3 = []
ilist4 = []
ilist5 = []

naccid = cogscores4[0][0]
for i in range(len(cogscores4)-1):
    if (cogscores4[i][0] != naccid):
        naccid = cogscores4[i][0]
    if (cogscores4[i+1][0] == naccid and i < 17000):
        timeint = cogscores4[i+1][1] - cogscores4[i][1]
        train = cogscores4[i, 2:75]
        train = np.insert(train, 0, timeint)
        label = cogscores4[i+1,56:75]
        thing = np.concatenate((train, label))
        ilist1.append(i+1)
        future_cog41.append(thing)
    if (cogscores4[i+1][0] == naccid and 17000 <= i < 34000):
        timeint = cogscores4[i+1][1] - cogscores4[i][1]
        train = cogscores4[i, 2:75]
        train = np.insert(train, 0, timeint)
        label = cogscores4[i+1,56:75]
        thing = np.concatenate((train, label))
        ilist2.append(i+1)
        future_cog42.append(thing)
    if (cogscores4[i+1][0] == naccid and 34000 <= i < 51000):
        timeint = cogscores4[i+1][1] - cogscores4[i][1]
        train = cogscores4[i, 2:75]
        train = np.insert(train, 0, timeint)
        label = cogscores4[i+1,56:75]
        thing = np.concatenate((train, label))
        ilist3.append(i+1)
        future_cog43.append(thing)
    if (cogscores4[i+1][0] == naccid and 51000 <= i < 68000):
        timeint = cogscores4[i+1][1] - cogscores4[i][1]
        train = cogscores4[i, 2:75]
        train = np.insert(train, 0, timeint)
        label = cogscores4[i+1,56:75]
        thing = np.concatenate((train, label))
        ilist4.append(i+1)
        future_cog44.append(thing)
    if (cogscores4[i+1][0] == naccid and 68000 <= i):
        timeint = cogscores4[i+1][1] - cogscores4[i][1]
        train = cogscores4[i, 2:75]
        train = np.insert(train, 0, timeint)
        label = cogscores4[i+1,56:75]
        thing = np.concatenate((train, label))
        ilist5.append(i+1)      
        future_cog45.append(thing)

ticker =  0
for i in range(len(future_cog41)):
    j = ilist1[i]
    if (np.count_nonzero(np.subtract(future_cog41[i][74:93], cogscores4[j, 56:75])) > 0):
        ticker+=1
print(ticker)

ticker =  0
for i in range(len(future_cog42)):
    j = ilist2[i]
    if (np.count_nonzero(np.subtract(future_cog42[i][74:93], cogscores4[j, 56:75])) > 0):
        ticker+=1
print(ticker)

ticker =  0
for i in range(len(future_cog43)):
    j = ilist3[i]
    if (np.count_nonzero(np.subtract(future_cog43[i][74:93], cogscores4[j, 56:75])) > 0):
        ticker+=1
print(ticker)

ticker =  0
for i in range(len(future_cog44)):
    j = ilist4[i]
    if (np.count_nonzero(np.subtract(future_cog44[i][74:93], cogscores4[j, 56:75])) > 0):
        ticker+=1
print(ticker)

ticker =  0
for i in range(len(future_cog45)):
    j = ilist5[i]
    if (np.count_nonzero(np.subtract(future_cog45[i][74:93], cogscores4[j, 56:75])) > 0):
        ticker+=1
print(ticker)
    
train1 = future_cog42+future_cog43+future_cog44+future_cog45
train2 = future_cog41+future_cog43+future_cog44+future_cog45
train3 = future_cog41+future_cog42+future_cog44+future_cog45
train4 = future_cog41+future_cog42+future_cog43+future_cog45
train5 = future_cog41+future_cog42+future_cog43+future_cog44

train1 = np.array(train1)
train2 = np.array(train2)
train3 = np.array(train3)
train4 = np.array(train4)
train5 = np.array(train5)

xend = 74
yend = 93

trainx1 = train1[:,:xend]
trainy1 = train1[:, xend:yend]
trainx2 = train2[:,:xend]
trainy2 = train2[:, xend:yend]
trainx3 = train3[:,:xend]
trainy3 = train3[:, xend:yend]
trainx4 = train4[:,:xend]
trainy4 = train4[:, xend:yend]
trainx5 = train5[:,:xend]
trainy5 = train5[:, xend:yend]


test1 = future_cog41
test2 = future_cog42
test3 = future_cog43
test4 = future_cog44
test5 = future_cog45

test1 = np.array(test1)
test2 = np.array(test2)
test3 = np.array(test3)
test4 = np.array(test4)
test5 = np.array(test5)
print(test5.shape)


testx1 = test1[:,:xend]
testy1 = test1[:, xend:yend]
testx2 = test2[:,:xend]
testy2 = test2[:, xend:yend]
testx3 = test3[:,:xend]
testy3 = test3[:, xend:yend]
testx4 = test4[:,:xend]
testy4 = test4[:, xend:yend]
testx5 = test5[:,:xend]
testy5 = test5[:, xend:yend]


trainx1 = np.array(trainx1)
trainy1 = np.array(trainy1)
trainx2 = np.array(trainx2)
trainy2 = np.array(trainy2)
trainx3 = np.array(trainx3)
trainy3 = np.array(trainy3)
trainx4 = np.array(trainx4)
trainy4 = np.array(trainy4)
trainx5 = np.array(trainx5)
trainy5 = np.array(trainy5)

testx1 = np.array(testx1)
testy1 = np.array(testy1)
testx2 = np.array(testx2)
testy2 = np.array(testy2)
testx3 = np.array(testx3)
testy3 = np.array(testy3)
testx4 = np.array(testx4)
testy4 = np.array(testy4)
testx5 = np.array(testx5)
testy5 = np.array(testy5)

epochs = 200
batch_size = 60
learning_rate = .01
learning_rate_decay = 0.96
concat = 1

errors_list1 = []
errors_list4 = []
falsepos4 = []
falseneg4 = []
errorsum = []
roundederrorsum = []

trainx1 = np.reshape(trainx1, (len(trainx1), 74, 1))
trainx2 = np.reshape(trainx2, (len(trainx2), 74, 1))
trainx3 = np.reshape(trainx3, (len(trainx3), 74, 1))
trainx4 = np.reshape(trainx4, (len(trainx4), 74, 1))
trainx5 = np.reshape(trainx5, (len(trainx5), 74, 1))

testx1 = np.reshape(testx1, (len(testx1), 74, 1))
testx2 = np.reshape(testx2, (len(testx2), 74, 1))
testx3 = np.reshape(testx3, (len(testx3), 74, 1))
testx4 = np.reshape(testx4, (len(testx4), 74, 1))
testx5 = np.reshape(testx5, (len(testx5), 74, 1))

traindatax = [trainx1, trainx2, trainx3, trainx4, trainx5]
traindatay = [trainy1, trainy2, trainy3, trainy4, trainy5]

testdatax = [testx1, testx2, testx3, testx4, testx5]
testdatay = [testy1, testy2, testy3, testy4, testy5]
    
print("######")
for i in range(5):
    for j in range(6):
        trmean = np.mean(traindatax[i][:, j])
        traindatax[i][:, j] = traindatax[i][:, j]-trmean
        traindatax[i][:, j] = traindatax[i][:, j]/np.abs(traindatax[i][:, j]).max(axis=0)
        temean = np.mean(testdatax[i][:, j])
        testdatax[i][:, j] = testdatax[i][:, j]-temean
        testdatax[i][:, j] = testdatax[i][:, j]/np.abs(testdatax[i][:, j]).max(axis=0)

def step_decay(epoch):
   initial_lrate = 0.01
   drop = 0.96
   epochs_drop = 200
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate
lrate = keras.callbacks.LearningRateScheduler(step_decay)
for i in range(5):
    print(i)
    model4 = Sequential()
    model4.add((Conv1D(128,1, activation = "relu", input_shape = (74, 1))))
    model4.add((Conv1D(64,1, activation = "relu")))
    model4.add((MaxPooling1D(1)))
    model4.add(LSTM(100, activation = "tanh", return_sequences = True))
    model4.add(LSTM(100, activation = "tanh"))
    model4.add(Dense(100, activation = "relu"))
    model4.add(Dense(19))
    optimizer = keras.optimizers.Adam(lr=0)
    model4.compile(loss='mae',optimizer=optimizer)
    history = model4.fit(x = traindatax[i], y =traindatay[i], batch_size = batch_size,
                         epochs=epochs,
                         validation_data=(testdatax[i], testdatay[i]),
                         callbacks = [lrate])
    print("########################")
    if (i==0):
        model4.save("/Users/Montague/Desktop/DStuff/naccrnn1")
    if (i==1):
        model4.save("/Users/Montague/Desktop/DStuff/naccrnn2")
    if (i==2):
        model4.save("/Users/Montague/Desktop/DStuff/naccrnn3")
    if (i==3):
        model4.save("/Users/Montague/Desktop/DStuff/naccrnn4")
    if (i==4):
        model4.save("/Users/Montague/Desktop/DStuff/naccrnn5")
    
    
    
    
    
    
    
    
    