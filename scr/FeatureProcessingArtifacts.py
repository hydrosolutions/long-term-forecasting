from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer, KNNImputer

logger = logging.getLogger(__name__)


import pickle
import joblib
import json
from pathlib import Path
import datetime


class FeatureProcessingArtifacts:
    """
    Container for all artifacts created during training data processing.
    These artifacts can be applied to new data for consistent preprocessing.
    """
    
    def __init__(self):
        self.imputer = None
        self.scaler = None
        self.long_term_means = None
        self.feature_selector = None
        self.selected_features = None
        self.highly_correlated_features = None
        self.final_features = None
        self.cat_features = None
        self.num_features = None
        self.creation_timestamp = None
        self.experiment_config_hash = None
        self.feature_count = None
        
    def save(self, filepath: Union[str, Path], format: str = 'joblib') -> None:
        """
        Save the artifacts to disk.
        
        Args:
            filepath: Path to save the artifacts (without extension)
            format: Format to use ('joblib', 'pickle', or 'hybrid')
        """
        filepath = Path(filepath)
        self.creation_timestamp = datetime.datetime.now().isoformat()
        self.feature_count = len(self.final_features) if self.final_features else 0
        
        if format == 'joblib':
            self._save_joblib(filepath)
        elif format == 'pickle':
            self._save_pickle(filepath)
        elif format == 'hybrid':
            self._save_hybrid(filepath)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'joblib', 'pickle', or 'hybrid'")
        
        logger.info(f"Artifacts saved to {filepath} using {format} format")
    
    def _save_joblib(self, filepath: Path) -> None:
        """Save using joblib (recommended for sklearn objects)."""
        filepath = filepath.with_suffix('.joblib')
        joblib.dump(self, filepath)
    
    def _save_pickle(self, filepath: Path) -> None:
        """Save using pickle."""
        filepath = filepath.with_suffix('.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    def _save_hybrid(self, filepath: Path) -> None:
        """
        Save using hybrid approach: JSON for metadata and dict objects, 
        joblib only for sklearn objects (safer than pickle).
        """
        # Create artifacts directory: parent_folder/artifacts_name/
        artifacts_dir = filepath.parent / filepath.name
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata as JSON
        metadata = {
            'selected_features': self.selected_features,
            'highly_correlated_features': self.highly_correlated_features,
            'final_features': self.final_features,
            'cat_features': self.cat_features,
            'num_features': self.num_features,
            'creation_timestamp': self.creation_timestamp,
            'experiment_config_hash': self.experiment_config_hash,
            'feature_count': self.feature_count
        }
        
        with open(artifacts_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save sklearn objects with joblib (only actual sklearn objects)
        sklearn_objects = {}
        if self.imputer is not None:
            sklearn_objects['imputer'] = self.imputer
        if self.feature_selector is not None:
            sklearn_objects['feature_selector'] = self.feature_selector
        
        if sklearn_objects:
            joblib.dump(sklearn_objects, artifacts_dir / "sklearn_objects.joblib")
        
        # Save scaler as JSON (it's a dict, not sklearn object)
        if self.scaler is not None:
            self._save_scaler_safe(artifacts_dir)
        
        # Save long_term_means as Parquet
        if self.long_term_means is not None:
            self._save_long_term_means_safe(artifacts_dir)
    
    def _save_scaler_safe(self, artifacts_dir: Path) -> None:
        """
        Save scaler dictionary in a safe, human-readable format.
        
        The scaler can be either:
        1. Global: {'feature_name': (mean, std), ...}
        2. Per-basin: {'basin_code': {'feature_name': (mean, std), ...}, ...}
        """
        if self.scaler is None:
            return
        
        # Convert scaler to JSON-serializable format
        scaler_data = {}
        
        for key, value in self.scaler.items():
            if isinstance(value, dict):
                # Per-basin case: key is basin_code, value is dict of feature scalers
                scaler_data[str(key)] = {}
                for feature, params in value.items():
                    if isinstance(params, (tuple, list)) and len(params) == 2:
                        scaler_data[str(key)][feature] = {
                            'mean': float(params[0]) if pd.notna(params[0]) else None,
                            'std': float(params[1]) if pd.notna(params[1]) else None
                        }
                    else:
                        logger.warning(f"Unexpected scaler format for {key}.{feature}: {params}")
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                # Global case: key is feature_name, value is (mean, std)
                scaler_data[str(key)] = {
                    'mean': float(value[0]) if pd.notna(value[0]) else None,
                    'std': float(value[1]) if pd.notna(value[1]) else None
                }
            else:
                logger.warning(f"Unexpected scaler format for {key}: {value}")
        
        # Save as JSON
        with open(artifacts_dir / "scaler.json", 'w') as f:
            json.dump(scaler_data, f, indent=2)
        
        logger.info(f"Saved scaler with {len(scaler_data)} entries as JSON")

    def _save_long_term_means_safe(self, artifacts_dir: Path) -> None:
        """
        Save long_term_means as a Parquet file.
        
        The long_term_means can be either:
        1. A pandas DataFrame
        2. A nested dict of DataFrames: {'basin_code': DataFrame, ...}
        3. A nested dict of dicts: {'basin_code': {'feature': value, ...}, ...}
        """
        if self.long_term_means is None:
            return
        
        try:
            # Case 1: Direct DataFrame
            if isinstance(self.long_term_means, pd.DataFrame):
                self.long_term_means.to_parquet(artifacts_dir / "long_term_means.parquet")
                logger.info("Saved long_term_means DataFrame as Parquet")
                return
            
            # Case 2 & 3: Nested structure
            if isinstance(self.long_term_means, dict):
                # Convert nested structure to a single DataFrame
                rows = []
                
                for basin_code, feature_data in self.long_term_means.items():
                    if isinstance(feature_data, pd.DataFrame):
                        # DataFrame case: add basin_code as a column
                        df_temp = feature_data.copy()
                        df_temp['code'] = basin_code
                        rows.append(df_temp)
                    elif isinstance(feature_data, pd.Series):
                        # Series case: convert to DataFrame
                        df_temp = feature_data.to_frame().T
                        df_temp['code'] = basin_code
                        rows.append(df_temp)
                    elif isinstance(feature_data, dict):
                        # Dict case: convert to DataFrame row
                        row_data = feature_data.copy()
                        row_data['code'] = basin_code
                        rows.append(pd.DataFrame([row_data]))
                    else:
                        logger.warning(f"Unexpected type for basin {basin_code}: {type(feature_data)}")
                
                if rows:
                    combined_df = pd.concat(rows, ignore_index=True)
                    combined_df.to_parquet(artifacts_dir / "long_term_means.parquet")
                    logger.info(f"Saved long_term_means for {len(self.long_term_means)} basins as Parquet")
                else:
                    logger.warning("No valid data found in long_term_means")
            
        except Exception as e:
            logger.error(f"Failed to save long_term_means as Parquet: {e}")
            # Fallback to JSON if Parquet fails
            logger.info("Falling back to JSON format")
            self._save_long_term_means_json_fallback(artifacts_dir)

    def _save_long_term_means_json_fallback(self, artifacts_dir: Path) -> None:
        """Fallback method to save as JSON if Parquet fails."""
        # ...existing code... (keep the original JSON implementation as fallback)
        flattened_means = {}
        
        for basin_code, feature_means in self.long_term_means.items():
            if isinstance(feature_means, dict):
                # Simple dict case
                for feature, mean_val in feature_means.items():
                    key = f"{basin_code}_{feature}"
                    flattened_means[key] = float(mean_val) if pd.notna(mean_val) else None
            elif hasattr(feature_means, 'to_dict'):
                # pandas Series case
                for feature, mean_val in feature_means.to_dict().items():
                    key = f"{basin_code}_{feature}"
                    flattened_means[key] = float(mean_val) if pd.notna(mean_val) else None
            else:
                logger.warning(f"Unexpected type for basin {basin_code}: {type(feature_means)}")
        
        # Save as JSON
        with open(artifacts_dir / "long_term_means.json", 'w') as f:
            json.dump(flattened_means, f, indent=2)
        
        logger.info("Saved long_term_means as JSON fallback")

    @classmethod
    def _load_hybrid(cls, filepath: Path) -> 'FeatureProcessingArtifacts':
        """Load from hybrid format with proper directory structure."""
        # Expect artifacts to be in: parent_folder/artifacts_name/
        artifacts_dir = filepath.parent / filepath.name
        
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
        
        artifacts = cls()
        
        # Load metadata
        metadata_path = artifacts_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            artifacts.selected_features = metadata.get('selected_features')
            artifacts.highly_correlated_features = metadata.get('highly_correlated_features')
            artifacts.final_features = metadata.get('final_features')
            artifacts.cat_features = metadata.get('cat_features')
            artifacts.num_features = metadata.get('num_features')
            artifacts.creation_timestamp = metadata.get('creation_timestamp')
            artifacts.experiment_config_hash = metadata.get('experiment_config_hash')
            artifacts.feature_count = metadata.get('feature_count')
        
        # Load sklearn objects
        sklearn_path = artifacts_dir / "sklearn_objects.joblib"
        if sklearn_path.exists():
            sklearn_objects = joblib.load(sklearn_path)
            artifacts.imputer = sklearn_objects.get('imputer')
            artifacts.feature_selector = sklearn_objects.get('feature_selector')
        
        # Load scaler from JSON
        artifacts.scaler = cls._load_scaler_safe(artifacts_dir)
        
        # Load long_term_means from Parquet or JSON
        artifacts.long_term_means = cls._load_long_term_means_safe(artifacts_dir)
        
        return artifacts
    
    @staticmethod
    def _load_scaler_safe(artifacts_dir: Path) -> Optional[Dict]:
        """Load scaler from JSON format."""
        scaler_path = artifacts_dir / "scaler.json"
        
        if not scaler_path.exists():
            return None
        
        with open(scaler_path, 'r') as f:
            scaler_data = json.load(f)
        
        # Convert back to the expected format
        scaler = {}
        
        for key, value in scaler_data.items():
            if isinstance(value, dict):
                if 'mean' in value and 'std' in value:
                    # Global case: convert back to tuple
                    scaler[key] = (value['mean'], value['std'])
                else:
                    # Per-basin case: convert nested dict
                    scaler[key] = {}
                    for feature, params in value.items():
                        if isinstance(params, dict) and 'mean' in params and 'std' in params:
                            scaler[key][feature] = (params['mean'], params['std'])
                        else:
                            logger.warning(f"Unexpected scaler format for {key}.{feature}: {params}")
            else:
                logger.warning(f"Unexpected scaler format for {key}: {value}")
        
        logger.info(f"Loaded scaler with {len(scaler)} entries from JSON")
        return scaler

    @staticmethod
    def _load_long_term_means_safe(artifacts_dir: Path) -> Optional[Dict]:
        """Load long_term_means from Parquet or JSON format."""
        parquet_path = artifacts_dir / "long_term_means.parquet"
        json_path = artifacts_dir / "long_term_means.json"
        
        # Try Parquet first
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                
                # Convert back to nested structure
                if 'code' in df.columns:
                    long_term_means = {}
                    
                    for basin_code in df['code'].unique():
                        basin_data = df[df['code'] == basin_code].drop('code', axis=1)
                        
                        if len(basin_data) == 1:
                            # Single row: convert to dict
                            long_term_means[basin_code] = basin_data.iloc[0].to_dict()
                        else:
                            # Multiple rows: keep as DataFrame
                            long_term_means[basin_code] = basin_data.reset_index(drop=True)
                    
                    logger.info(f"Loaded long_term_means for {len(long_term_means)} basins from Parquet")
                    return long_term_means
                else:
                    # Direct DataFrame case
                    logger.info("Loaded long_term_means DataFrame from Parquet")
                    return df.to_dict()
                    
            except Exception as e:
                logger.warning(f"Failed to load Parquet file: {e}")
                logger.info("Falling back to JSON format")
        
        # Fallback to JSON
        if json_path.exists():
            with open(json_path, 'r') as f:
                flattened_means = json.load(f)
            
            # Reconstruct the nested structure
            long_term_means = {}
            
            for key, value in flattened_means.items():
                if '_' not in key:
                    logger.warning(f"Unexpected key format: {key}")
                    continue
                    
                # Split basin_code and feature_name
                parts = key.split('_', 1)  # Split only on first underscore
                basin_code, feature_name = parts[0], parts[1]
                
                if basin_code not in long_term_means:
                    long_term_means[basin_code] = {}
                
                long_term_means[basin_code][feature_name] = value
            
            logger.info(f"Loaded long_term_means for {len(long_term_means)} basins from JSON")
            return long_term_means
        
        return None
    @classmethod
    def load(cls, filepath: Union[str, Path], format: str = 'auto') -> 'FeatureProcessingArtifacts':
        """
        Load artifacts from disk.
        
        Args:
            filepath: Path to the artifacts file
            format: Format to use ('auto', 'joblib', 'pickle', or 'hybrid')
            
        Returns:
            Loaded FeatureProcessingArtifacts instance
        """
        filepath = Path(filepath)
        
        if format == 'auto':
            format = cls._detect_format(filepath)
        
        if format == 'joblib':
            return cls._load_joblib(filepath)
        elif format == 'pickle':
            return cls._load_pickle(filepath)
        elif format == 'hybrid':
            return cls._load_hybrid(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @staticmethod
    def _detect_format(filepath: Path) -> str:
        """Auto-detect the format based on file extensions and directory structure."""
        # Check for hybrid format (directory structure)
        artifacts_dir = filepath.parent / filepath.name
        if artifacts_dir.exists() and (artifacts_dir / "metadata.json").exists():
            return 'hybrid'
        
        # Check for single file formats
        if filepath.with_suffix('.joblib').exists():
            return 'joblib'
        elif filepath.with_suffix('.pkl').exists():
            return 'pickle'
        else:
            raise FileNotFoundError(f"No artifacts found at {filepath}")

    @classmethod
    def _load_joblib(cls, filepath: Path) -> 'FeatureProcessingArtifacts':
        """Load from joblib file."""
        filepath = filepath.with_suffix('.joblib')
        return joblib.load(filepath)

    @classmethod
    def _load_pickle(cls, filepath: Path) -> 'FeatureProcessingArtifacts':
        """Load from pickle file."""
        filepath = filepath.with_suffix('.pkl')
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the artifacts."""
        return {
            'creation_timestamp': self.creation_timestamp,
            'feature_count': self.feature_count,
            'has_imputer': self.imputer is not None,
            'has_scaler': self.scaler is not None,
            'has_long_term_means': self.long_term_means is not None,
            'has_feature_selector': self.feature_selector is not None,
            'final_features': self.final_features,
            'num_features_count': len(self.num_features) if self.num_features else 0,
            'cat_features_count': len(self.cat_features) if self.cat_features else 0
        }
    



def process_training_data(
    df_train: pd.DataFrame,
    features: List[str],
    target: str,
    experiment_config: Dict[str, Any],
    pca_groups: Optional[Dict[str, List[str]]] = None,
    variance_threshold: float = 0.95
) -> Tuple[pd.DataFrame, FeatureProcessingArtifacts]:
    """
    Process training data and create artifacts for consistent preprocessing.
    
    Args:
        df_train: Training DataFrame
        features: List of feature names
        target: Target variable name
        experiment_config: Configuration dictionary containing processing parameters
        pca_groups: Optional PCA grouping configuration
        variance_threshold: Variance threshold for PCA
        
    Returns:
        Tuple of (processed_training_data, artifacts)
    """
    artifacts = FeatureProcessingArtifacts()
    df_processed = df_train.copy()
    
    # Separate numeric and categorical features early
    
    artifacts.num_features = [col for col in features if df_train[col].dtype.kind in 'ifc']
    artifacts.cat_features = [col for col in features if col not in artifacts.num_features]
    
    logger.info(f"Numeric features: {len(artifacts.num_features)}")
    logger.info(f"Categorical features: {len(artifacts.cat_features)}")
    
    # 1. Handle missing values and create imputation artifacts
    df_processed, artifacts = _handle_missing_values_training(
        df_processed, features, target, experiment_config, artifacts
    )
    
    # 2. Feature selection and correlation removal
    df_processed, artifacts = _feature_selection_training(
        df_processed, target, experiment_config, artifacts
    )

    # 3. Normalization (create scaler)
    df_processed, artifacts = _normalization_training(
        df_processed, target, experiment_config, artifacts
    )
    
    # 4. PCA (if configured) - placeholder for now
    if pca_groups:
        logger.warning("PCA functionality not implemented in this refactored version")
    
    # Set final features
    artifacts.final_features = artifacts.selected_features if artifacts.selected_features else features
    
    logger.info(f"Final feature count: {len(artifacts.final_features)}")

    return df_processed, artifacts
    
def save_artifacts_for_production(
    artifacts: FeatureProcessingArtifacts,
    model_name: str,
    version: str,
    base_path: Union[str, Path] = "./models"
) -> Path:
    """
    Save artifacts for production use with versioning and metadata.
    
    Args:
        artifacts: FeatureProcessingArtifacts instance
        model_name: Name of the model
        version: Version string (e.g., "v1.0.0", "2024-01-15")
        base_path: Base directory for saving models
        
    Returns:
        Path to saved artifacts
    """
    base_path = Path(base_path)
    model_dir = base_path / model_name / version
    model_dir.mkdir(parents=True, exist_ok=True)
    
    artifact_path = model_dir / "feature_artifacts"
    
    # Add version info to artifacts
    artifacts.model_name = model_name
    artifacts.version = version
    
    # Save using hybrid format for production robustness
    artifacts.save(artifact_path, format='hybrid')
    
    # Create a latest symlink for easy access
    latest_path = base_path / model_name / "latest"
    if latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(version, target_is_directory=True)
    
    logger.info(f"Production artifacts saved: {artifact_path}")
    logger.info(f"Latest symlink updated: {latest_path}")
    
    return artifact_path


def load_artifacts_for_production(
    model_name: str,
    version: str = "latest",
    base_path: Union[str, Path] = "./models"
) -> FeatureProcessingArtifacts:
    """
    Load artifacts for production use.
    
    Args:
        model_name: Name of the model
        version: Version to load ("latest" for most recent)
        base_path: Base directory where models are stored
        
    Returns:
        Loaded FeatureProcessingArtifacts instance
    """
    base_path = Path(base_path)
    model_dir = base_path / model_name / version
    artifact_path = model_dir / "feature_artifacts"
    
    if not artifact_path.exists() and not (artifact_path.parent / f"{artifact_path.name}_metadata.json").exists():
        raise FileNotFoundError(f"No artifacts found for {model_name} version {version}")
    
    artifacts = FeatureProcessingArtifacts.load(artifact_path)
    
    logger.info(f"Loaded artifacts for {model_name} {version}")
    logger.info(f"Artifacts created: {artifacts.creation_timestamp}")
    logger.info(f"Feature count: {artifacts.feature_count}")
    
    return artifacts


def process_test_data(
    df_test: pd.DataFrame,
    artifacts: FeatureProcessingArtifacts,
    experiment_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply preprocessing artifacts to test data for consistent processing.
    
    Args:
        df_test: Test DataFrame
        artifacts: Artifacts created during training data processing
        experiment_config: Configuration dictionary
        
    Returns:
        Processed test DataFrame
    """
    df_processed = df_test.copy()
    
    # Apply the same preprocessing steps using training artifacts
    df_processed = _apply_missing_value_handling(
        df=df_processed, artifacts=artifacts, experiment_config=experiment_config)
    df_processed = _apply_feature_selection(
        df=df_processed, artifacts=artifacts)
    df_processed = _apply_normalization(
        df=df_processed, artifacts=artifacts, experiment_config=experiment_config)
    
    return df_processed


def process_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: List[str],
    target: str,
    experiment_config: Dict[str, Any],
    pca_groups: Optional[Dict[str, List[str]]] = None,
    variance_threshold: float = 0.95
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], FeatureProcessingArtifacts]:
    """
    Wrapper function that processes both training and test data consistently.
    
    Args:
        df_train: Training DataFrame
        df_test: Test DataFrame
        features: List of feature names
        target: Target variable name
        experiment_config: Configuration dictionary containing processing parameters
        pca_groups: Optional PCA grouping configuration
        variance_threshold: Variance threshold for PCA
        
    Returns:
        Tuple of (processed_train, processed_test, final_features, artifacts)
    """
    # Process training data and create artifacts
    df_train_processed, artifacts = process_training_data(
        df_train, features, target, experiment_config, pca_groups, variance_threshold
    )
    
    # Apply artifacts to test data
    df_test_processed = process_test_data(df_test, artifacts, experiment_config)
    
    return df_train_processed, df_test_processed, artifacts.final_features, artifacts


# Helper functions for training phase
def _handle_missing_values_training(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    experiment_config: Dict[str, Any],
    artifacts: FeatureProcessingArtifacts
) -> Tuple[pd.DataFrame, FeatureProcessingArtifacts]:
    """Handle missing values and create imputation artifacts."""
    
    handle_na = experiment_config.get('handle_na', 'drop')
    
    if handle_na == 'drop':
        all_cols = list(set(features + [target]))
        df = df.dropna(subset=all_cols)
        logger.info("Applied dropna strategy")
        
    elif handle_na == 'long_term_mean':
        # Import here to avoid circular imports
        from scr import data_utils as du
        numeric_features = [col for col in features if df[col].dtype.kind in 'ifc']
        
        artifacts.long_term_means = du.get_long_term_mean_per_basin(df, features=numeric_features)
        df = du.apply_long_term_mean(df, long_term_mean=artifacts.long_term_means, features=numeric_features)
        logger.info("Created long-term mean artifacts")
        logger.info(f"Long Term Means: {artifacts.long_term_means}")

    elif handle_na == 'impute':
        impute_cols = [col for col in features if df[col].dtype.kind in 'ifc']
        
        if impute_cols and df[impute_cols].isna().any().any():
            impute_method = experiment_config.get('impute_method', 'mean')
            
            if impute_method == 'knn':
                artifacts.imputer = KNNImputer(n_neighbors=5)
            else:
                artifacts.imputer = SimpleImputer(strategy=impute_method)
            
            # Fit and transform training data
            imputed_data = artifacts.imputer.fit_transform(df[impute_cols])
            df[impute_cols] = pd.DataFrame(imputed_data, columns=impute_cols, index=df.index)
            logger.info(f"Created imputer with strategy: {impute_method}")
    
    return df, artifacts


def _feature_selection_training(
    df: pd.DataFrame,
    target: str,
    experiment_config: Dict[str, Any],
    artifacts: FeatureProcessingArtifacts
) -> Tuple[pd.DataFrame, FeatureProcessingArtifacts]:
    """Perform feature selection and create selection artifacts."""
    
    if not experiment_config.get('use_mutual_info', False):
        artifacts.selected_features = artifacts.num_features + artifacts.cat_features
        return df, artifacts
    
    X_train = df[artifacts.num_features]
    y_train = df[target]
    
    # Remove highly correlated features
    if experiment_config.get('remove_correlated_features', False):
        corr_matrix = X_train.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        artifacts.highly_correlated_features = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > 0.95)
        ]
        X_train = X_train.drop(columns=artifacts.highly_correlated_features)
        logger.info(f"Identified highly correlated features: {artifacts.highly_correlated_features}")
    
    # Select best features using mutual information
    n_features = experiment_config.get('number_of_features', 10)
    artifacts.feature_selector = SelectKBest(mutual_info_regression, k=n_features)
    artifacts.feature_selector.fit(X_train, y_train)
    
    selected_numeric_features = X_train.columns[artifacts.feature_selector.get_support()].tolist()
    logger.info(f"Selected features from mutual information: {selected_numeric_features}")
    
    # Combine with categorical features
    artifacts.selected_features = list(set(selected_numeric_features) | set(artifacts.cat_features))
    
    return df, artifacts


def _normalization_training(
    df: pd.DataFrame,
    target: str,
    experiment_config: Dict[str, Any],
    artifacts: FeatureProcessingArtifacts
) -> Tuple[pd.DataFrame, FeatureProcessingArtifacts]:
    """Perform normalization and create scaler artifacts."""
    
    if not experiment_config.get('normalize', False):
        return df, artifacts
    
    # Import here to avoid circular imports
    from scr import data_utils as du
    
    # Get numeric features from selected features
    numeric_features_to_scale = [
        col for col in artifacts.selected_features 
        if col in artifacts.num_features
    ]
    
    if experiment_config.get('normalize_per_basin', False):
        artifacts.scaler = du.get_normalization_params_per_basin(
            df, numeric_features_to_scale, target
        )  
        df = du.apply_normalization_per_basin(
            df, artifacts.scaler, numeric_features_to_scale
        )
    else:
        artifacts.scaler = du.get_normalization_params(
            df, numeric_features_to_scale, target
        )
        df = du.apply_normalization(
            df, artifacts.scaler, numeric_features_to_scale
            )

    logger.info("Created normalization scaler")
    return df, artifacts


# Helper functions for applying artifacts to test data
def _apply_missing_value_handling(
    df: pd.DataFrame,
    artifacts: FeatureProcessingArtifacts,
    experiment_config: Dict[str, Any]
) -> pd.DataFrame:
    """Apply missing value handling using training artifacts."""
    
    handle_na = experiment_config.get('handle_na', 'drop')
    
    if handle_na == 'drop':
        # For test data, we might want to handle this differently
        # Could drop rows with missing values in selected features only
        if artifacts.final_features:
            df = df.dropna(subset=artifacts.final_features)
        
    elif handle_na == 'long_term_mean' and artifacts.long_term_means is not None:
        from scr import data_utils as du
        df = du.apply_long_term_mean(
            df, 
            long_term_mean=artifacts.long_term_means, 
            features=artifacts.num_features
        )
        
    elif handle_na == 'impute' and artifacts.imputer is not None:
        impute_cols = [col for col in artifacts.num_features if col in df.columns]
        
        if impute_cols and df[impute_cols].isna().any().any():
            imputed_data = artifacts.imputer.transform(df[impute_cols])
            df[impute_cols] = pd.DataFrame(imputed_data, columns=impute_cols, index=df.index)
    
    return df


def _apply_feature_selection(
    df: pd.DataFrame,
    artifacts: FeatureProcessingArtifacts
) -> pd.DataFrame:
    """Apply feature selection using training artifacts."""
    
    # Remove highly correlated features if identified during training
    if artifacts.highly_correlated_features:
        cols_to_drop = [col for col in artifacts.highly_correlated_features if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
    
    # Note: Feature selector is already applied through the selected_features list
    # The actual column selection happens when we use artifacts.final_features
    
    return df


def _apply_normalization(
    df: pd.DataFrame,
    artifacts: FeatureProcessingArtifacts,
    experiment_config: Dict[str, Any]
) -> pd.DataFrame:
    """Apply normalization using training artifacts."""
    
    if not experiment_config.get('normalize', False) or artifacts.scaler is None:
        return df
    
    # Import here to avoid circular imports
    from scr import data_utils as du
    
    # Get numeric features that were normalized during training
    numeric_features_to_scale = [
        col for col in artifacts.selected_features 
        if col in artifacts.num_features and col in df.columns
    ]
    
    if experiment_config.get('normalize_per_basin', False):
        df = du.apply_normalization_per_basin(df, artifacts.scaler, numeric_features_to_scale)
    else:
        df = du.apply_normalization(df, artifacts.scaler, numeric_features_to_scale)
    
    return df



def post_process_target(
    df_predictions: pd.DataFrame,
    target: str,
    artifacts: FeatureProcessingArtifacts,
    experiment_config: Dict[str, Any],
    prediction_column: str = 'Q_pred'
    ) -> pd.DataFrame:

    """
    Simple implementation - reverse normalization of predictions using artifacts.
    
    Args:
        df_predictions: DataFrame with predictions
        target: Target variable name  
        artifacts: FeatureProcessingArtifacts containing scaler
        experiment_config: Configuration dictionary
        prediction_column: Name of the prediction column to denormalize
        
    Returns:
        DataFrame with denormalized predictions
    """
    df_predictions = df_predictions.copy()
    
    if not experiment_config.get('normalize', False) or artifacts.scaler is None:
        return df_predictions
    
    if experiment_config.get('normalize_per_basin', False):
        # Basin-specific denormalization
        if 'code' not in df_predictions.columns:
            logger.warning("Basin code column not found for per-basin denormalization")
            return df_predictions
            
        for code in df_predictions['code'].unique():
            if code in artifacts.scaler and target in artifacts.scaler[code]:
                mean_, std_ = artifacts.scaler[code][target]
                mask = df_predictions['code'] == code
                df_predictions.loc[mask, prediction_column] = (
                    df_predictions.loc[mask, prediction_column] * std_ + mean_
                )
            else:
                logger.warning(f"No scaler found for basin {code} and target {target}")
    else:
        # Global denormalization
        if target in artifacts.scaler:
            mean_, std_ = artifacts.scaler[target]
            df_predictions[prediction_column] = (
                df_predictions[prediction_column] * std_ + mean_
            )
        else:
            logger.warning(f"No scaler found for target {target}")
    
    logger.info(f"Applied denormalization to {prediction_column}")
    return df_predictions