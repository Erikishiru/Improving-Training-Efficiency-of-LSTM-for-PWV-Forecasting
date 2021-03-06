# Improving-Training-Efficiency-of-LSTM-for-PWV-Forecasting

With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript:

> Jain, M., Yadav, P., Wu, J. and Dev, S.(2021). "Improving Training Efficiency of LSTMs while Forecasting Precipitable Water Vapours." Progress in Electromagnetic Research Symposium, Hangzhou, China. 2021.

# Dependencies

```
numpy
json (included)
pickle (included)
matplotlib
tensorflow
```

# Constants

| Variable             | Use Case                                                                       |
| -------------------- | :----------------------------------------------------------------------------- |
| NAME                 | A name tag for the model file for logging and creating individual result files |
| data_history         | Past data to be considered while creating the windows                          |
| futureStepToPredict  | Lead time for every prediction                                                 |
| batch_size           | Batch size for training                                                        |
| data_split           | Test-Train data split for training                                             |
| tf.random.set_seed() | Random seed for TensorFlow for constant results while training                 |
| np.random.seed()     | Random seed for Numpy for constant results while training                      |

# Functions

| Functions                                                                                                                                           | Description                                                                                  |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------- |
| load_obj(name)                                                                                                                                      | Load the data file passed as `name`                                                          |
| check_leap_year(year)                                                                                                                               | checks if the year is a leap year to change the number of days while making data continuous  |
| makeContinuousWithNANs(pwvComplete)                                                                                                                 | Make data continuous by adding NANs in the place of missing data in the passed PWV data      |
| makeWindowsFromContinuousSet(continuous_years, continuous_days, continuous_hours, continuous_minutes, continuous_pwv, history, futureStepToPredict) | Make Data Windows for LSTM training using enumerations created in `makeContinuousWithNANs()` |
| extractPWVonly(listOfDataWithTimeInfo)                                                                                                              | Extract PWV value column from the complete dataset                                           |
| reshape_feature_data(trainX, testX)                                                                                                                 | Reshape Feature data for model training                                                      |
| np.random.seed()                                                                                                                                    | Random seed for Numpy for constant results while training                                    |

# Folder structure

| Variable           | Use Case                               |
| ------------------ | :------------------------------------- |
| data               | To store the data files used           |
| SavedModels        | To save checkpoints and trained models |
| Graphs_and_Results | To store Graphs and Results            |
