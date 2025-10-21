# TODO 

Glacier Mapper feature show not much improvemnt so far - just one month -> june.
Run the linear regression again without ridge and see if it improves (same setting as Snowmapper)
if not - use ridge again
run GBT LR fro glacier mapper again

run combined gbt - add most imporant gla mapper features

use linear regression where i first predict delta NIR = NIR - wX(Q,T,P) 
this should then be the unexplainable part by the other vraibles , see if it improves.

### Decide on Models
- [x] Base Case LR (1)
- [x] Base Case GBT (3)
- [x] SM LR, 
- [x] SM LR DT, 
- [x] SM LR ROF 
- [x] SM GBT (3)
- [x] SM GBT LR (3)
- [x] SM GBT Norm (3)
- [ ] SM GBT Elev (3)
- [x] Uncertainty ALD
  
This would result in 19 models...
How many max features for the linear regression models? 2, 3, 4 (i think it is very unlikely that 4 features have a high correlation - set to 4)? it is also coupled with a correlation threshold (set to 0.4 so far).

## Operationalize
- [ ] Integration of Meta-Model in Prediction Pipeline
- [ ] Integration of Uncertainty Net in Prediction Pipeline
- [ ] Decide on Models
- [ ] Release Version 0.1.0

## Fixes 
- [ ]

## Features (Needed)
- [ ] Plot Generation (workshop)
- [ ] Documentation

## Features (Nice to Have)
- [ ] Deep Learning Regressors