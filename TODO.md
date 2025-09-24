# TODO 

## Next steps

### Push
I put all not used files in Z folders (deep models) - run the uncertainty mixture model to see if if it still works - then create a pull/merge request. -> should work.

### Uncertainty MLP
- [ ] create a plot showing the individual memebers pdf and the resulting combination for sanity checking.
- [ ] validate on metrics like coverage , obs vs theretical prob of exceedance, cprs etc.
- [ ] Create operational prediction

### Decide on Models
- [ ] Base Case LR (1)
- [ ] Base Case GBT (3)
- [ ] SM LR, 
- [ ] SM LR DT, 
- [ ] SM LR ROF 
- [ ] SM GBT (3)
- [ ] SM GBT LR (3)
- [ ] SM GBT Norm (3)
- [ ] SM GBT Elev (3)
  
This would result in 19 models...
How many max features for the linear regression models? 2, 3, 4 (i think it is very unlikely that 4 features have a high correlation - set to 4)? it is also coupled with a correlation threshold (set to 0.4 so far).

## Urgent Fixes
- [ ] Integration of Meta-Model in Prediction Pipeline
- [x] Uncertainty Net - Calculate error statistics for all years except the one at hand - loocv errors -> this creates new columns in the data frame - error_mean, error_skew, error_std, error_max , abs_error_mean, abs_error_std, abs_error_max
- [ ] Test model performance


## Fixes 
- [x] Proper Scaling -> Improved performance


## Features (Needed)
- [x] Uncertainty Net
- [ ] Plot Generation (workshop)

## Features (Nice to Have)
- [ ] Deep Learning Regressors