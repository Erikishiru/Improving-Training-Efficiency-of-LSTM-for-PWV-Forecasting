"""**Description**

Custom LSTM implementation with PWV dataset
Lead time of 1 step in the future
With Custom Loss  with l=0.05, k=10
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

NAME = 'custom_lstm_2month_1step_v2' #name of the python file for logging

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

dataX = dataX[0 : 8640*2]
dataY = dataY[0 : 8640*2]

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
time = np.arange(0,8000)
plt.plot(continuous_pwv[time])
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

"""### Custom Model Defination with loss function """

loss_tracker = Mean(name="loss")
rmse_metric = RootMeanSquaredError(name='rmse')

class CustomModel(Model):
    def train_step(self, data):
        x, y = data

        with GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            def my_regularizer_loss(y_pred, x_latest, y_true):
                l = 0.05
                k = 10
                y_pred = tf.cast(y_pred, dtype=K.floatx())
                x_latest = tf.cast(x_latest, dtype=K.floatx())
                error = tf.subtract(y_pred, x_latest[:,-1,:])
                abs_error = tf.abs(error)
                my_k = tf.convert_to_tensor(k, dtype=abs_error.dtype)
                my_l = tf.convert_to_tensor(l, dtype=abs_error.dtype)
                sqr_part = tf.pow(tf.abs(my_k*abs_error), 2)
                # exp_part = tf.exp(-my_k*abs_error)
                # abs_naive_diff = tf.abs(tf.subtract(y_true, x_latest[:,-1,:]))
                # my_thresh = tf.convert_to_tensor(thresh, dtype=abs_naive_diff.dtype)
                # return K.mean(tf.multiply(tf.multiply(l, abs_naive_diff), exp_part))
                return K.mean(tf.multiply(l, sqr_part))

            # loss = tf.where(
            #                 loss_switch[0,0] > tf.constant(0.0),
            #                 huber(y, y_pred) + my_regularizer_loss(y_pred, x, y),
            #                 huber(y, y_pred)
            #                 )
            loss = huber(y, y_pred) + tf.minimum(my_regularizer_loss(y_pred, x, y), 3.0)
            loss = tf.where (
                loss > tf.constant(2.0), loss, huber(y, y_pred)
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        rmse_metric.update_state(y, y_pred)
        return {"loss": loss_tracker.result(), "rmse": rmse_metric.result()}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Update the metrics
        rmse_metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {"rmse": rmse_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker, rmse_metric]

import os
from os import path, mkdir

result_dir_path = './Graphs_and_Results/' + NAME
model_dir_path = './SavedModels/'+ NAME

if not path.exists(result_dir_path):
    os.mkdir(result_dir_path)

if not path.exists(model_dir_path):
    os.mkdir(model_dir_path)

"""### LR Schedule Test"""

K.clear_session()

# Construct an instance of CustomModel
input_layer = Input((data_history, 1))
lstm_1 = LSTM(32, activation='sigmoid', return_sequences=True)(input_layer)
lstm_2 = LSTM(32, activation='sigmoid')(lstm_1)
output_layer = Dense(1)(lstm_2)
model = CustomModel(input_layer, output_layer)

lr_schedule = LearningRateScheduler(
    lambda epoch: 1e-4 * 10**(epoch / 20))

lr_schedule_json_log = open('./Graphs_and_Results/' + NAME + '/lr_schedule_log.json', mode='wt', buffering=1)

lr_schedule_log = LambdaCallback(
  on_epoch_end = lambda epoch, logs: lr_schedule_json_log.write(
        json.dumps({'epoch': str(epoch), 'loss': str(logs['loss']), 'lr': str(logs['lr']), 'rmse': str(logs['rmse'])}) + '\n'),
    on_train_end=lambda logs: lr_schedule_json_log.close()
)

checkpointer = ModelCheckpoint(filepath='./SavedModels/'+ NAME +'/lrSchedule_weights.hdf5', monitor="loss", verbose=1, save_best_only=True, save_weights_only=True)
adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer="adam")
print(model.summary())
print("Starting to fit")
history = model.fit(trainXpwv_reshaped, trainYpwv, epochs=100, batch_size=batch_size, verbose=1, callbacks=[lr_schedule, checkpointer, lr_schedule_log])

model.load_weights('./SavedModels/' + NAME + '/lrSchedule_weights.hdf5')

plt.figure(figsize=(20, 12))
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-6, 1e+5, 0.1, 55.0])
plt.xlabel('Learning Rate')
plt.ylabel('Loss')

plt.savefig('./Graphs_and_Results/' + NAME + '/lrSchedule_test.png', bbox_inches='tight')

"""#### Training the model"""

K.clear_session()

# Construct an instance of CustomModel
input_layer = Input((data_history, 1))
lstm_1 = LSTM(32, activation='sigmoid',return_sequences=True)(input_layer)
lstm_2 = LSTM(32, activation='sigmoid')(lstm_1)
output_layer = Dense(1)(lstm_2)
model = CustomModel(input_layer, output_layer)

lr_schedule = LearningRateScheduler(
    lambda epoch: 1e-4 * 10**(epoch / 20) if (1e-4 * 10**(epoch / 20)<1e-2) else 1e-2)

training_json_log = open('./Graphs_and_Results/' + NAME + '/training_log.json', mode='wt', buffering=1)

training_log = LambdaCallback(
  on_epoch_end = lambda epoch, logs: training_json_log.write(
        json.dumps({'epoch': str(epoch), 'loss': str(logs['loss']), 'lr': str(logs['lr']), 'rmse': str(logs['rmse'])}) + '\n'),
    on_train_end=lambda logs: training_json_log.close()
)

checkpointer = ModelCheckpoint(filepath='./SavedModels/'+ NAME +'/train_weights.hdf5', monitor="loss", verbose=1, save_best_only=True, save_weights_only=True)
adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer="adam")
print(model.summary())
print("Starting to fit with loss switch on")
history = model.fit(trainXpwv_reshaped, trainYpwv, epochs=150, batch_size=batch_size, verbose=1, callbacks=[lr_schedule, checkpointer, training_log])

model.load_weights('./SavedModels/' + NAME + '/train_weights.hdf5')

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

plt.savefig('./Graphs_and_Results/' + NAME + '/train_loss.png', bbox_inches='tight')

print(training_loss)

"""### Evaluating the Model"""

# model = tf.keras.models.load_model('./SavedModels/' + NAME + '/model.hdf5')
# Evaluate Model
print('Training Data Evaluation')
print(model.evaluate(x = trainXpwv_reshaped, y = trainYpwv))
print('Test Data Evaluation')
print(model.evaluate(x = testXpwv_reshaped, y = testYpwv))

model.save('./SavedModels/' + NAME + '/model.hdf5')

"""### Model Predictions """

def naiveForecast(x_data):
    # Expected Shape of x_data: (number of samples, number of readings per sample, 1)
    # Shape of forecast: (number of samples, )
    forecast = x_data[:,-1,0]
    return forecast

naive_forecast = naiveForecast(testXpwv_reshaped)
print(naive_forecast.shape)
forecast = model.predict(testXpwv_reshaped)
print(forecast.shape)

from sklearn.metrics import mean_squared_error
from math import sqrt

rms_lstm = sqrt(mean_squared_error(testYpwv, forecast))
print("Root Mean Squared Error Vanilla wrt LSTM = ", rms_lstm)
rms_naive = sqrt(mean_squared_error(testYpwv, naive_forecast))
print("Root Mean Squared Error wrt Naive = ", rms_naive)

plt.figure(figsize=(20, 12))
time = np.arange(1000,2000)
plt.plot(time,testYpwv[time])
plt.plot(time,naive_forecast[time])
plt.plot(time,forecast[time])
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(("Actual Data", "Naive Forecast", "Forecasted Data"))
plt.grid(True)

plt.savefig('./Graphs_and_Results/' + NAME + '/predictions.png', bbox_inches='tight')

plt.figure(figsize=(20, 12))
time = np.arange(1250,1400)
plt.plot(time,testYpwv[time])
plt.plot(time,naive_forecast[time])
plt.plot(time,forecast[time])
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(("Actual Data", "Naive Forecast", "Forecasted Data"))
plt.grid(True)

plt.savefig('./Graphs_and_Results/' + NAME + '/zoomed_predictions.png', bbox_inches='tight')
