# TODO 

## Next steps
Currently working on the uncertainty_mixture model. 
The fit mmethod is not fully implemented -> needs fitting and using lightning data classes.
Each ensemble member produces a mu, sigma and tau parameter for asymetric laplace 
so it needs a combination model -> mixture model 
where each memeber is weighted and the density functions are added togheter (so it is still a valid pdf). 
-> write an extra class or function for this.
Question is how to deal with missing ensemble memebers.

calibrate and hindcast needs to be implemented

predict operational needs to be implemented

tune hyperparameter needs to be implemented 

this should be fairly straight forward when the fit and predict function is in place

- [ ] create a plot showing the individual memebers pdf and the resulting combination for sanity checking.
- [ ] validate on metrics like coverage , obs vs theretical prob of exceedance, cprs etc.


## Urgent Fixes
- [ ] Integration of Meta-Model in Prediction Pipeline

## Fixes 
- [x] Proper Scaling -> Improved performance


## Features (Needed)
- [ ] Deep Learning Meta-Learner / Uncertainty Net
- [ ] Plot Generation

## Features (Nice to Have)
- [ ] Deep Learning Regressors