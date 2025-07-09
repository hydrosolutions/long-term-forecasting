# Source Code (scr) Directory

## Purpose
Core utility modules providing data handling, feature extraction, and processing capabilities for the monthly forecasting system.

## Contents
- `__init__.py`: Package initialization
- `data_loading.py`: Data ingestion and loading utilities
- `data_utils.py`: Data manipulation and transformation functions
- `FeatureExtractor.py`: Advanced feature engineering for tree-based models
- `FeatureProcessingArtifacts.py`: Feature processing state management
- `sci_utils.py`: Scientific computing utilities
- `tree_utils.py`: Tree-based model specific utilities

## Important Classes and Functions

### data_loading.py
- `load_training_data()`: Loads training datasets from specified paths
- `load_test_data()`: Loads test/validation datasets
- Handles multiple data formats (CSV, Parquet)
- Manages missing data and date parsing

### data_utils.py
- `apply_preprocessing()`: Applies various preprocessing methods
- `scale_data()`: Data normalization and scaling
- `handle_missing_values()`: Missing data imputation
- Date/time feature extraction utilities

### FeatureExtractor.py
- `FeatureExtractor` class: Advanced feature engineering
  - Lag features creation
  - Rolling statistics (mean, std, min, max)
  - Seasonal decomposition
  - Cross-feature interactions
  - Temporal features (month, season, year)

### FeatureProcessingArtifacts.py
- `FeatureProcessingArtifacts` class: Manages feature processing state
  - Stores preprocessing parameters
  - Maintains feature scaling information
  - Tracks feature importance
  - Enables reproducible preprocessing

### sci_utils.py
- Scientific computing utilities
- Statistical functions
- Time series analysis helpers
- Custom metrics implementation

### tree_utils.py
- Tree-specific preprocessing
- Feature importance extraction
- Model-specific data formatting
- Ensemble method utilities

## Key Features
- Modular design for easy extension
- Comprehensive error handling
- Efficient data processing pipelines
- Support for large datasets
- Preprocessing method flexibility

## Usage Examples
```python
# Data loading
from scr.data_loading import load_training_data
train_data = load_training_data(path="data/train.csv")

# Feature extraction
from scr.FeatureExtractor import FeatureExtractor
extractor = FeatureExtractor(config)
features = extractor.extract_features(data)

# Preprocessing
from scr.data_utils import apply_preprocessing
processed_data = apply_preprocessing(data, method="monthly_bias")
```

## Integration Points
- Used by all model implementations
- Connected to evaluation pipeline
- Integrated with test suite
- Referenced by calibration scripts