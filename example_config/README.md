# Example Configuration Files

This directory contains example configuration files for the monthly discharge forecasting system. These files serve as templates for setting up new experiments and models.

## Directory Structure

- `DUMMY_MODEL/` - Example configuration for a dummy model setup

## Configuration Files

Each model experiment requires the following configuration files:

### 1. `data_paths.json`
Defines paths to input data files:
- `path_to_discharge`: Observed discharge data
- `path_to_forcing`: Temperature and precipitation data
- `path_to_static_basin_characteristics`: Basin attributes
- `path_to_snow_data`: Snow water equivalent (SWE), height of snow (HS), runoff (ROF)
- `path_to_sca`: Snow cover area data
- `path_to_sla`: Snow line altitude from GlacierMapper (optional)

### 2. `experiment_config.json`
Defines experiment setup:
- `experiment_name`: Unique identifier for the experiment
- `basins`: List of basin codes to include
- `data_split`: Train/test split configuration
- `calibration_period`: Date range for model training
- `evaluation_period`: Date range for model evaluation

### 3. `feature_config.json`
Specifies feature engineering parameters:
- `time_windows`: Rolling window sizes for feature extraction
- `operations`: Statistical operations (mean, slope, peak-to-peak, etc.)
- `feature_selections`: Which feature types to include
- `lag_periods`: Number of lag periods to consider

### 4. `general_config.json`
Contains general model settings:
- `model_name`: Model identifier
- `normalization_type`: Type of data normalization
- `use_long_term_mean_scaling`: Enable/disable period-based scaling
- `num_elevation_zones`: Number of elevation bands to use
- `glacier_mapper_features_to_keep`: List of GlacierMapper features to retain
- `use_glacier_features`: Enable/disable glacier-related features

### 5. `model_config.json`
Model-specific hyperparameters:
- Algorithm-specific parameters (e.g., XGBoost, LightGBM, CatBoost)
- Ensemble configuration
- Feature selection thresholds
- Training parameters

## Usage

1. Copy the `DUMMY_MODEL` directory as a template:
   ```bash
   cp -r example_config/DUMMY_MODEL config/MY_NEW_MODEL
   ```

2. Modify the configuration files according to your experiment needs

3. Run the model with your configuration:
   ```bash
   uv run calibrate_hindcast.py --config_path config/MY_NEW_MODEL
   ```

## Best Practices

- Keep experiment names descriptive and unique
- Document any special configuration choices in comments
- Test configuration validity before running long experiments
- Version control your configuration files for reproducibility