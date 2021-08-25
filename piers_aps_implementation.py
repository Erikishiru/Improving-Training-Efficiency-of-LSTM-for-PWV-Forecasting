# -*- coding: utf-8 -*-
"""**Description**

LSTM implementation with PWV dataset
Lead time of 1 step in the future
Without Custom Loss with l=0
"""

"""# **Import Required Libraries**"""

import numpy as np
import json
import pickle
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, LambdaCallback
from tensorflow.keras.losses import huber, mse
import tensorflow.keras.backend as K

"""### Defining constants"""

NAME = 'vanilla_lstm_sequential_1step' #name of the python file for logging

data_history = 47 #past data to be considered for a window

futureStepToPredict = 1 # Lead time for every prediction

batch_size = 500

data_split = 0.8

tf.random.set_seed(51)
np.random.seed(51)

"""# **Read and Pre-process Data**
**Read Data**
"""

def load_obj(name):
  with open(name + '.pkl', 'rb') as f:
    return pickle.load(f)

data_file = './data/pwvComplete5yr'
pwvComplete = load_obj(data_file) # Dictionary with year numbers as keys and numpy arrays ([day, hour, min, pwv]) as values

"""**Pre-process Data**"""

def check_leap_year(year):
  if year%4==0:
    return True
  else:
    return False

# Make data continuous by adding NANs in the place of missing data
def makeContinuousWithNANs(pwvComplete):
  resolutionMinutes = 5
  readingsPerHour = int(60/resolutionMinutes)
  hoursPerDay = int(24)
  readingsPerDay = int(hoursPerDay*readingsPerHour)

  yearwise_days     = np.zeros((len(pwvComplete), 366*readingsPerDay))
  yearwise_hours    = np.zeros((len(pwvComplete), 366*readingsPerDay))
  yearwise_minutes  = np.zeros((len(pwvComplete), 366*readingsPerDay))
  yearwise_pwv      = np.zeros((len(pwvComplete), 366*readingsPerDay))
  continuous_years  = np.empty(0)
  continuous_days   = np.empty(0)
  continuous_hours  = np.empty(0)
  continuous_minutes= np.empty(0)
  continuous_pwv    = np.empty(0)
  prev_year = 0
  for yearID, year in enumerate(sorted(pwvComplete.keys())):
    if not check_leap_year(int(year)):
      nDays = 365
    else:
      nDays = 366
    new_days    = np.concatenate([(i+1)*np.ones((readingsPerDay)) for i in range(0, nDays)], axis=None)
    new_hours   = np.concatenate([(j)*np.ones((readingsPerHour)) for i in range(0, nDays) for j in range(0, 24)], axis=None)
    new_minutes = np.concatenate([k for i in range(0, nDays) for j in range(0, 24) for k in range(0, 60, resolutionMinutes)], axis=None)
    new_pwv     = np.empty(nDays*readingsPerDay)
    new_pwv[:]  = np.nan

    counter = 0
    for idx in range(0, nDays*readingsPerDay):
      if (new_days[idx] == pwvComplete[year][0][counter]) and (new_hours[idx] == pwvComplete[year][1][counter]) and (new_minutes[idx] == pwvComplete[year][2][counter]):
        new_pwv[idx] = pwvComplete[year][3][counter]
        counter += 1
    yearwise_days[yearID, :nDays*readingsPerDay]    = new_days
    yearwise_hours[yearID, :nDays*readingsPerDay]   = new_hours
    yearwise_minutes[yearID, :nDays*readingsPerDay] = new_minutes
    yearwise_pwv[yearID, :nDays*readingsPerDay]     = new_pwv

    if (not prev_year == 0) and (not (int(year)-prev_year) == 1):
      while not (int(year)-prev_year == 1):
        prev_year += 1
        continuous_years    = np.concatenate((continuous_years, np.array([prev_year])))
        continuous_days     = np.concatenate((continuous_days, np.array([1])))
        continuous_hours    = np.concatenate((continuous_hours, np.array([0])))
        continuous_minutes  = np.concatenate((continuous_minutes, np.array([0])))
        continuous_pwv      = np.concatenate((continuous_pwv, np.array([np.nan])))
    continuous_years    = np.concatenate((continuous_years, int(year)*np.ones((nDays*readingsPerDay))))
    continuous_days     = np.concatenate((continuous_days, new_days))
    continuous_hours    = np.concatenate((continuous_hours, new_hours))
    continuous_minutes  = np.concatenate((continuous_minutes, new_minutes))
    continuous_pwv      = np.concatenate((continuous_pwv, new_pwv))
    prev_year = int(year)
  return (yearwise_days, yearwise_hours, yearwise_minutes, yearwise_pwv, continuous_years, continuous_days, continuous_hours, continuous_minutes, continuous_pwv)

# Make Data Windows - Note: history includes the value at current value (at t=0) too
def makeWindowsFromContinuousSet(continuous_years, continuous_days, continuous_hours, continuous_minutes, continuous_pwv, history, futureStepToPredict=1):
  dataX, dataY = [], []
  continuousSubSequenceStartIndex = 0
  continuousSubSequenceStopIndex  = 0
  searchingForNAN = True
  seqCounter = 0
  while seqCounter<len(continuous_pwv):
    if not np.isnan(continuous_pwv[seqCounter]):
      if not searchingForNAN:
        searchingForNAN = True
        continuousSubSequenceStartIndex = seqCounter
    else:
      if searchingForNAN:
        searchingForNAN = False
        continuousSubSequenceStopIndex = seqCounter - 1
        ### Perform continuousSubSequence Operation - Extract Windows ###
        if continuousSubSequenceStopIndex - continuousSubSequenceStartIndex + 1 >= history + futureStepToPredict:
          for i in range(continuousSubSequenceStartIndex, continuousSubSequenceStopIndex+1):
            if i + history + futureStepToPredict <= continuousSubSequenceStopIndex + 1:
              dataX.append(np.vstack((continuous_years[i:(i+history)],
                                      continuous_days[i:(i+history)],
                                      continuous_hours[i:(i+history)],
                                      continuous_minutes[i:(i+history)],
                                      continuous_pwv[i:(i+history)])))
              dataY.append(np.vstack((continuous_years[i+history+futureStepToPredict-1],
                                      continuous_days[i+history+futureStepToPredict-1],
                                      continuous_hours[i+history+futureStepToPredict-1],
                                      continuous_minutes[i+history+futureStepToPredict-1],
                                      continuous_pwv[i+history+futureStepToPredict-1])))
    seqCounter += 1
  return dataX, dataY

def extractPWVonly(listOfDataWithTimeInfo):
  pwvValues = []
  for i in range(len(listOfDataWithTimeInfo)):
    pwvValues.append(listOfDataWithTimeInfo[i][4,:])
  return pwvValues

(yearwise_days, yearwise_hours, yearwise_minutes, yearwise_pwv,
 continuous_years, continuous_days, continuous_hours, continuous_minutes, continuous_pwv) = makeContinuousWithNANs(pwvComplete)

#removing outliers
continuous_pwv[(5 > continuous_pwv) | (continuous_pwv > 100)] = np.nan

#taking only 2010 data for training
dataX, dataY = makeWindowsFromContinuousSet(continuous_years[continuous_years == 2010], continuous_days[continuous_years == 2010], continuous_hours[continuous_years == 2010], continuous_minutes[continuous_years == 2010], continuous_pwv[continuous_years == 2010],
                                            history=data_history, futureStepToPredict=futureStepToPredict)
print(len(dataX), dataX[0].shape)
print(len(dataY), dataY[0].shape)

train_split = data_split
split_time = int(train_split*len(dataY))
trainX = dataX[:split_time]
trainY = dataY[:split_time]
testX = dataX[split_time:]
testY = dataY[split_time:]
print(len(trainX), len(trainY), len(testX), len(testY))
trainXpwv = np.array(extractPWVonly(trainX))
trainYpwv = np.array(extractPWVonly(trainY))
testXpwv  = np.array(extractPWVonly(testX))
testYpwv  = np.array(extractPWVonly(testY))
print(trainXpwv.shape, trainYpwv.shape, testXpwv.shape, testYpwv.shape)

plt.figure(figsize=(20, 12))
time = np.arange(16000,17000)
plt.plot(continuous_pwv)
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)

"""# **Train Forecasting Models**"""

def reshape_feature_data(trainX, testX):
  print("Before Reshaping")
  print(trainX.shape)
  print(testX.shape)

  trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
  testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

  print("After Reshaping")
  print(trainX.shape)
  print(testX.shape)

  return trainX, testX

trainXpwv_reshaped, testXpwv_reshaped = reshape_feature_data(trainXpwv, testXpwv)

"""### LR Schedule Test"""

model = Sequential()
model.add(LSTM(32, input_shape=(data_history, 1), activation='sigmoid', return_sequences=True))
model.add(LSTM(32, activation='sigmoid'))
model.add(Dense(1))

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-4 * 10**(epoch / 20))

lr_schedule_json_log = open('./Graphs_and_Results/' + NAME + '_lr_schedule_log.json', mode='wt', buffering=1)

lr_schedule_log = LambdaCallback(
  on_epoch_end = lambda epoch, logs: lr_schedule_json_log.write(
        json.dumps({'epoch': str(epoch), 'loss': str(logs['loss']), 'lr': str(logs['lr']), 'mean_squared_error': str(logs['mean_squared_error'])}) + '\n'),
    on_train_end=lambda logs: lr_schedule_json_log.close()
)

checkpointer = ModelCheckpoint(filepath='./SavedModels/'+ NAME +'_lrSchedule_weights.hdf5', monitor="loss", verbose=1, save_best_only=True, save_weights_only=True)
adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss="huber_loss",
            optimizer=adam,
            metrics=["mean_squared_error"])
print(model.summary())

print("Starting to fit")
history = model.fit(trainXpwv_reshaped, trainYpwv, epochs=100, batch_size=batch_size, verbose=1, callbacks=[lr_schedule, checkpointer, lr_schedule_log])

model.load_weights('./SavedModels/' + NAME + '_lrSchedule_weights.hdf5')

plt.figure(figsize=(20, 12))
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-6, 1e+5, 0.1, 55.0])
plt.xlabel('Learning Rate')
plt.ylabel('Loss')

plt.savefig('./Graphs_and_Results/' + NAME + '_lrSchedule_test.png', bbox_inches='tight')

"""#### Training the model"""

model = Sequential()
model.add(LSTM(32, input_shape=(data_history, 1), activation='sigmoid', return_sequences=True))
model.add(LSTM(32, activation='sigmoid'))
model.add(Dense(1))

lr_schedule = LearningRateScheduler(
    lambda epoch: 1e-4 * 10**(epoch / 20) if (1e-4 * 10**(epoch / 20)<1e-2) else 1e-2)

training_json_log = open('./Graphs_and_Results/' + NAME + '_training_log.json', mode='wt', buffering=1)

training_log = LambdaCallback(
  on_epoch_end = lambda epoch, logs: training_json_log.write(
        json.dumps({'epoch': str(epoch), 'loss': str(logs['loss']), 'lr': str(logs['lr']), 'mean_squared_error': str(logs['mean_squared_error'])}) + '\n'),
    on_train_end=lambda logs: training_json_log.close()
)

checkpointer = ModelCheckpoint(filepath='./SavedModels/'+ NAME +'_train_weights.hdf5', monitor="loss", verbose=1, save_best_only=True, save_weights_only=True)
adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss="huber_loss",
            optimizer=adam,
            metrics=["mean_squared_error"])
print(model.summary())

print("Starting to fit")
history = model.fit(trainXpwv_reshaped, trainYpwv, epochs=150, batch_size=batch_size, verbose=1, callbacks=[lr_schedule, checkpointer, training_log])

model.load_weights('./SavedModels/' + NAME + '_train_weights.hdf5')

# Get training and test loss histories
training_loss = history.history['loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.figure(figsize=(10, 6))
plt.plot(epoch_count, training_loss, 'r--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.savefig('./Graphs_and_Results/' + NAME + '_train_loss.png', bbox_inches='tight')

"""### Evaluating the Model"""

# Evaluate Model
print('Training Data Evaluation')
print(model.evaluate(x = trainXpwv_reshaped, y = trainYpwv))
print('Test Data Evaluation')
print(model.evaluate(x = testXpwv_reshaped, y = testYpwv))

model.save('./SavedModels/' + NAME + '_model.hdf5')

"""### Model Predictions """

forecast = model.predict([testXpwv_reshaped, testYpwv])
print(forecast.shape)

plt.figure(figsize=(20, 12))
time = np.arange(16000,17000)
plt.plot(time,testYpwv[time])
plt.plot(time,forecast[time])
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(("Actual Data","Forecasted Data"))
plt.grid(True)

plt.savefig('./Graphs_and_Results/' + NAME + '_predictions.png', bbox_inches='tight')

plt.figure(figsize=(20, 12))
time = np.arange(16520,16600)
plt.plot(time,testYpwv[time])
plt.plot(time,forecast[time])
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(("Actual Data","Forecasted Data"))
plt.grid(True)

plt.savefig('./Graphs_and_Results/' + NAME + '_zoomed_predictions.png', bbox_inches='tight')