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
from keras.regularizers import l2


from sklearn.model_selection import KFold

cogscores4 = np.load("onehotmissimpute.npy", allow_pickle = True)

np.set_printoptions(suppress=True)

#bills, taxes, shopping, games, stove, 5+5+5+5+5 = 25
#mealprep, events, payattn, remdates, 5+5+5+5 = 20 (45)
#travel, memory, orient, judgment, commun,  5+5+5+5+5 = 25 (70)
#home, perscare, cdrsum, cdrlang, comport, cdrglob 5+4+32+5+5+5 = 56 (126)

future_cog_x = []
future_cog_y = []

ilist = []

naccid = cogscores4[0][0]
end = len(cogscores4[0])
start = end-126
for i in range(len(cogscores4)-1):
    if (cogscores4[i][0] != naccid):
        naccid = cogscores4[i][0]
    if (cogscores4[i+1][0] == naccid):
        timeint = cogscores4[i+1][1] - cogscores4[i][1]
        train = cogscores4[i, 2:end]
        train = np.insert(train, 0, timeint)
        label = cogscores4[i+1,start:end]
        ilist.append(i+1)
        future_cog_x.append(train)
        future_cog_y.append(label)
        
ticker =  0
for i in range(len(future_cog_y)):
    j = ilist[i]
    if (np.count_nonzero(np.subtract(future_cog_y[i], cogscores4[j, start:end])) > 0):
        ticker+=1
print(ticker)
        
X = np.array(future_cog_x)
y = np.array(future_cog_y)

kf = KFold(n_splits=5)

epochs = 200
batch_size = 60
learning_rate = .01
learning_rate_decay = 0.96
concat = 1

def step_decay(epoch):
   initial_lrate = 0.01
   drop = 0.96
   epochs_drop = 200
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

lrate = keras.callbacks.LearningRateScheduler(step_decay)

percenterrorlist = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train = np.reshape(X_train, (len(X_train), end-1, 1))
    X_test = np.reshape(X_test, (len(X_test), end-1, 1))
    model = Sequential()
    model.add(LSTM(128, activation = "softmax", return_sequences = True, kernel_regularizer=l2(0.0001)))
    model.add(LSTM(128, activation = "softmax", kernel_regularizer=l2(0.0001)))
    model.add(Dense(126, activation = "relu", kernel_regularizer=l2(0.0001)))
    optimizer = keras.optimizers.Adam(lr=0)
    model.compile(loss='binary_crossentropy',optimizer=optimizer)
    history = model.fit(x = X_train, y = y_train, batch_size = batch_size,
                         epochs=epochs,
                         validation_data=(X_test, y_test),
                         callbacks = [lrate])
    print("########################")
    predicted = model.predict(X_test)
    predicted = np.rint(predicted)
    summy = np.add(predicted, y_test)
    oldnonzero = np.count_nonzero(y_test)
    newnonzero = np.count_nonzero(summy)
    percentage = (newnonzero-oldnonzero)/oldnonzero
    print(percentage)
    percenterrorlist.append(percentage)
          
          
          
          
          
          
          
          
          
          
          
          