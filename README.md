# Monthly Discharge Forecasting
Author: Sandro Hunziker


## General Concept (general concept)

The structure of this code is following a modular approach. There are different forecasting classes which all adhere and provide similar functions. These classes are not bound to a specific model but rather a type of implementation. For example: the linear regression workflow is packed into one class and the workflow for tree-based models and scikit learn models (SVR, MLP etc..). 
These classes handle the data processing part and therefore one can easyily provide small ensembles. For example if a XGB, LGBM and Catboost model all use the same inputs, we only need to process the data once and can then predict and calibrate all models at the same time. This enables us to create fast a bunch of ensembles. 

We can have for example a set of model (lets say XGB, LGBM, CatBoost) with features $F_1 \isin (Q, P, T)$, $F_2 \isin (Q, T, P, Snowmapper FSM)$, $F_3 \isin (GlacierMapper, Q, T, P)$ leading to 9 different model. We can then also extend the feature space to integreate longer lockback windows, creating one set of features $F_1, F_2, F_3 \text{ (with short term  feature focus)}$ and $F_4, F_5, F_6 \text{ (with long-term feature focus)}$ (leading to an ensemble of 18 models). Operationally, one can then show only the ensemble mean for the different feature sets. 

Of course, this produces a computational overhead as we each time fit 3 models, but the fitting process is quiet fast and this ensembling improves forecast accuracy.



 ## Folder Structure 
 - monthly_forecasting
   - scr
     - __init__.py
     - data_loading.py
     - data_utils.py
     - FeatureExtractor.py
     - tree_utils.py
   - forecast_models
     - base_class.py
     - LINEAR_REGRESSION.py
     - SciRegressor.py
   - eval_scr
     - __init__.py
     - metric_functions.py
     - eval_helper.py
   - meta_scr
     - ..some helper scripts for meta model
   - calibrate_hindcast.py
   - tune_hyperparams.py
   - run_operational_forecast.py
   - hindcast_calibrate.sh
 - monthly_forecasting_models
   - config
     - data_paths.json
     - base_learner_paths.json
   - Ensemble_XY
     - feature_config.json
     - data_config.json
     - general_config.json
     - hyperparams_config.json
     - scalers (csv)
   - LinearRegression_Q_T_P
     - feature_config.json
     - data_config.json
     - general_config.json
   - LinearRegression_Q_SCA
     - ....
   - LinearRegression_Q_SCA_T_P
     - ...
 - monthly_forecasting_results
   - Ensemble_XY
     - predictions.csv
     - metrics.csv



## Methods

### Calibration & Validation

The yearly leave-one-out cross-validation is used on all of the years except the last 3 available years. Those are left out as a final test set. So the predictions.csv set of each model is the prediction on these left out years + the test years. For meta-learning and cascade like models, we assume that those LOO-CV prediction represent how the model behave on unseen data. 

### Base Learner Models

We use a set of periodic Linear Regression Models. Predictos include features based on past discharge, precipitation, temperature and snow information from Snowmapper FSM based on different elevation zones. 
For the tree based models we can create a bunch of possible features also based on discharge, precipitation, temperature and snow information from Snowmapper FSM (lumped). Additionally we can use data from GlacierMapper as a earth observation based data source. Some tree based models can also take the predictions from the linear regressions as an additional input.

### Ensemble and Meta-Model

1. For the naive ensemble we have use just the naive mean of all the base predictos (Ensemble Mean).
2. Use a temporal meta-model - which should detect sharp drifts and changes from single model and ingore those - adjust final prediction based on past forecasts and some observations.
3. Uncertainty net: Uses context and the base-learner predictions to introduce a prediction interval (following the asymetric-laplace distribution). 


## Output format

$\textbf{predictions.csv}$ \
date | Q_model1 | Q_model2 | Q_model3 | Q_mean | valid_from | valid_to

Q_model corresponds to the prediction of a ensmeble member and Q_mean is the average over these models. \

$\textbf{predictions.csv}$  (for the meta model with uncertainty)\
date | Q_05| Q_10 | Q_50 | Q_90 | Q_95 | Q_mean | valid_from | valid_to


## Feature Sets

### Linear Regressions

\begin{table}
    \begin{tabular}[c|ccc]
    name & name short & 
    \end{tabular}
\end{table}

Name Long, Name Short, Available Features

