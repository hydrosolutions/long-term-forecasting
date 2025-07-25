# What Needs to Be Changed

In the deep scr is the code base for a meta_learning model based on a DL model.

The following needs to be changed.

## DataSet

The data set needs to be more generic - providing the input types:
at prediction time t:
past time steps: t-lookback : t
future_time_steps: t:t+future_steps
now_data : t

x_past (batch, past_time_steps, past_features) -> such as past discharge , P , T , past predictions of models..
x_nan_mask (batch, past_time_steps, past_features) -> binary mask which feature is missing
x_future (batch, future_time_steps: future_vars): weather forecast, week_sin, week_cos etc
x_now: (batch, 1, now_vars) current data for today (predictions from other models, historical errors from other models at time t)
x_static: static features

The data set needs to create a nan mask for the past feature and fills the nan values with 0. (if flag nan_mask = True) other wise it drops items with nan in input


## Model

For now the models are standalone implementation but I want a class for deep learning models, which can either be meta learners or standalone forecast models. They should inherit the base forecasting model
and do the same steps as the SciRegressor.
1. Init
   1. initialize loss
   2. hyperparams etc
2. data preprocessing 
   1. Loading everything
   2. Calculating the target
   3. Defineing the features
   4. Loading base_predictors
   5. Retransformation functions
3. LOOCV (yearly basis)
   1. for each year create a train and test set (test starts at the test_year - loockback days to ensure we can fully predict the year), the last 0.1 percentage of the train set is used for early stopping
   2. Predict the test year
4. Fit on all
   1. Use 0.9 percentage of the data to fit a model and 0.1 for early stopping
   2. Save the scalers used for this run
   3. Save the model for this run
5. Predict Operational
   1. Load the models and scalers
   2. Preprocess the data exactly as in loocv mode
   3. Predict - ensure that all codes get a prediction - set to nan if not possible
6. Hparam tuning
   1. Tune few hparams on a defined train / test set 
   2. Hidden size, learning rate, dropout, weight decay

see train_eval.py script for reference (this script is not good but just a refernce)

## Structure of the Deep learning based models

in deep_scr create a folder models/ here we define the torch basd architectures which can then be accessed by the main forecasting class

## Models wanted

1. AL Uncertainty - similar model as now (belongs to the meta learner) uses the mean of the base_models as mode of the assymetric laplace distribution and just predicts the associated uncertainty with it.
2. DeepForecaster - deep forecasting model which predicts directly the output (+uncertainty) either trained on mse loss or asymetric laplace loss - backbones may include LSTM, CNN-LSTM, Mamba , TiDE, TSMixer 
   

   

