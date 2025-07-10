"""
Main evaluation pipeline for monthly discharge forecasting.

This script orchestrates the complete evaluation workflow:
1. Load all predictions
2. Create ensemble predictions
3. Evaluate all models (individual + ensembles)
4. Generate outputs for dashboard consumption
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import argparse
import sys

# Import evaluation modules
from prediction_loader import load_all_predictions, validate_prediction_data
from ensemble_builder import create_all_ensembles, save_ensemble_predictions, combine_predictions_and_ensembles
from evaluate_models import evaluate_multiple_models, calculate_model_rankings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EvaluationPipeline:
    """
    Main evaluation pipeline class that orchestrates the complete workflow.
    """
    
    def __init__(self, 
                 results_dir: str = "../monthly_forecasting_results",
                 output_dir: str = "../monthly_forecasting_results/evaluation",
                 evaluation_day: Union[str, int] = 'end',
                 common_codes_only: bool = True,
                 ensemble_method: str = 'mean',
                 include_code_month: bool = False,
                 min_samples_overall: int = 10,
                 min_samples_code: int = 5,
                 min_samples_month: int = 3,
                 min_samples_code_month: int = 2):
        """
        Initialize the evaluation pipeline.
        
        Parameters:
        -----------
        results_dir : str
            Directory containing model results
        output_dir : str
            Directory to save evaluation outputs
        evaluation_day : Union[str, int]
            Day of month for evaluation ('end' or integer)
        common_codes_only : bool
            Whether to use only common basin codes
        ensemble_method : str
            Method for ensemble creation
        include_code_month : bool
            Whether to include per-code-month evaluation
        min_samples_overall : int
            Minimum samples for overall evaluation
        min_samples_code : int
            Minimum samples for per-code evaluation
        min_samples_month : int
            Minimum samples for per-month evaluation
        min_samples_code_month : int
            Minimum samples for per-code-month evaluation
        """
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.evaluation_day = evaluation_day
        self.common_codes_only = common_codes_only
        self.ensemble_method = ensemble_method
        self.include_code_month = include_code_month
        self.min_samples_overall = min_samples_overall
        self.min_samples_code = min_samples_code
        self.min_samples_month = min_samples_month
        self.min_samples_code_month = min_samples_code_month
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.loaded_predictions = {}
        self.validation_results = {}
        self.family_ensembles = {}
        self.global_ensembles = {}
        self.all_predictions = {}
        self.evaluation_results = pd.DataFrame()
        self.metadata = {}
        
        logger.info("Evaluation pipeline initialized")
    
    def load_predictions(self) -> bool:
        """
        Load all prediction data.
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            logger.info("=== STEP 1: Loading Predictions ===")
            
            self.loaded_predictions, self.validation_results = load_all_predictions(
                results_dir=self.results_dir,
                evaluation_day=self.evaluation_day,
                common_codes_only=self.common_codes_only
            )
            
            if not self.loaded_predictions:
                logger.error("No predictions loaded successfully")
                return False
            
            logger.info(f"Successfully loaded {len(self.loaded_predictions)} models")
            
            # Log validation summary
            total_warnings = sum(len(v.get('warnings', [])) for v in self.validation_results.values())
            if total_warnings > 0:
                logger.warning(f"Found {total_warnings} validation warnings")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading predictions: {str(e)}")
            return False
    
    def create_ensembles(self) -> bool:
        """
        Create family and global ensemble predictions.
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            logger.info("=== STEP 2: Creating Ensembles ===")
            
            self.family_ensembles, self.global_ensembles = create_all_ensembles(
                loaded_predictions=self.loaded_predictions,
                ensemble_method=self.ensemble_method,
                create_global=True
            )
            
            logger.info(f"Created {len(self.family_ensembles)} family ensembles")
            logger.info(f"Created {len(self.global_ensembles)} global ensembles")
            
            # Save ensemble predictions
            saved_files = save_ensemble_predictions(
                family_ensembles=self.family_ensembles,
                global_ensembles=self.global_ensembles,
                output_dir=self.output_dir
            )
            
            logger.info(f"Saved {len(saved_files)} ensemble files")
            
            # Combine all predictions
            self.all_predictions = combine_predictions_and_ensembles(
                loaded_predictions=self.loaded_predictions,
                family_ensembles=self.family_ensembles,
                global_ensembles=self.global_ensembles
            )
            
            logger.info(f"Combined total: {len(self.all_predictions)} models/ensembles")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating ensembles: {str(e)}")
            return False
    
    def evaluate_models(self) -> bool:
        """
        Evaluate all models and ensembles.
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            logger.info("=== STEP 3: Evaluating Models ===")
            
            self.evaluation_results = evaluate_multiple_models(
                loaded_predictions=self.all_predictions,
                include_code_month=self.include_code_month,
                min_samples_overall=self.min_samples_overall,
                min_samples_code=self.min_samples_code,
                min_samples_month=self.min_samples_month,
                min_samples_code_month=self.min_samples_code_month
            )
            
            if self.evaluation_results.empty:
                logger.error("No evaluation results generated")
                return False
            
            logger.info(f"Generated {len(self.evaluation_results)} evaluation records")
            
            # Calculate evaluation level counts
            level_counts = self.evaluation_results['level'].value_counts()
            for level, count in level_counts.items():
                logger.info(f"  {level}: {count} records")
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating models: {str(e)}")
            return False
    
    def generate_outputs(self) -> bool:
        """
        Generate all output files for dashboard consumption.
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            logger.info("=== STEP 4: Generating Outputs ===")
            
            output_path = Path(self.output_dir)
            
            # 1. Save comprehensive metrics
            metrics_file = output_path / "metrics.csv"
            self.evaluation_results.to_csv(metrics_file, index=False)
            logger.info(f"Saved comprehensive metrics to {metrics_file}")
            
            # 2. Generate metrics summary
            summary_stats = self._generate_metrics_summary()
            summary_file = output_path / "metrics_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            logger.info(f"Saved metrics summary to {summary_file}")
            
            # 3. Generate family comparison
            family_metrics = self._generate_family_metrics()
            family_file = output_path / "model_family_metrics.csv"
            family_metrics.to_csv(family_file, index=False)
            logger.info(f"Saved family metrics to {family_file}")
            
            # 4. Generate model rankings
            rankings = self._generate_model_rankings()
            rankings_file = output_path / "model_rankings.csv"
            rankings.to_csv(rankings_file, index=False)
            logger.info(f"Saved model rankings to {rankings_file}")
            
            # 5. Generate evaluation metadata
            self._generate_evaluation_metadata()
            metadata_file = output_path / "evaluation_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            logger.info(f"Saved evaluation metadata to {metadata_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating outputs: {str(e)}")
            return False
    
    def _generate_metrics_summary(self) -> Dict:
        """Generate summary statistics for all metrics."""
        summary = {}
        
        # Overall metrics summary
        overall_results = self.evaluation_results[self.evaluation_results['level'] == 'overall']
        
        if not overall_results.empty:
            metrics = ['r2', 'rmse', 'nrmse', 'mae', 'mape', 'nse', 'kge', 'bias', 'pbias']
            
            for metric in metrics:
                if metric in overall_results.columns:
                    summary[metric] = {
                        'mean': overall_results[metric].mean(),
                        'std': overall_results[metric].std(),
                        'min': overall_results[metric].min(),
                        'max': overall_results[metric].max(),
                        'median': overall_results[metric].median(),
                        'count': overall_results[metric].count()
                    }
        
        # Family performance summary
        family_performance = {}
        for family in overall_results['family'].unique():
            family_data = overall_results[overall_results['family'] == family]
            if not family_data.empty:
                family_performance[family] = {
                    'n_models': len(family_data),
                    'mean_r2': family_data['r2'].mean(),
                    'mean_nse': family_data['nse'].mean(),
                    'mean_kge': family_data['kge'].mean()
                }
        
        summary['family_performance'] = family_performance
        
        return summary
    
    def _generate_family_metrics(self) -> pd.DataFrame:
        """Generate family-level comparison metrics."""
        overall_results = self.evaluation_results[self.evaluation_results['level'] == 'overall']
        
        if overall_results.empty:
            return pd.DataFrame()
        
        # Group by family and calculate statistics
        family_stats = overall_results.groupby('family').agg({
            'r2': ['mean', 'std', 'min', 'max', 'count'],
            'nse': ['mean', 'std', 'min', 'max'],
            'kge': ['mean', 'std', 'min', 'max'],
            'nrmse': ['mean', 'std', 'min', 'max'],
            'mape': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        # Flatten column names
        family_stats.columns = [f"{col[1]}_{col[0]}" for col in family_stats.columns]
        family_stats = family_stats.reset_index()
        
        return family_stats
    
    def _generate_model_rankings(self) -> pd.DataFrame:
        """Generate model rankings for different metrics."""
        rankings_list = []
        
        metrics = ['r2', 'nse', 'kge', 'nrmse', 'mape']
        
        for metric in metrics:
            if metric in self.evaluation_results.columns:
                ascending = metric in ['nrmse', 'mape', 'bias', 'pbias']  # Lower is better
                
                ranking = calculate_model_rankings(
                    evaluation_results=self.evaluation_results,
                    metric=metric,
                    level='overall',
                    ascending=ascending
                )
                
                if not ranking.empty:
                    ranking['ranking_metric'] = metric
                    rankings_list.append(ranking)
        
        if rankings_list:
            return pd.concat(rankings_list, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _generate_evaluation_metadata(self):
        """Generate evaluation metadata."""
        self.metadata = {
            'pipeline_version': '1.0.0',
            'evaluation_date': datetime.now().isoformat(),
            'configuration': {
                'results_dir': self.results_dir,
                'output_dir': self.output_dir,
                'evaluation_day': self.evaluation_day,
                'common_codes_only': self.common_codes_only,
                'ensemble_method': self.ensemble_method,
                'include_code_month': self.include_code_month,
                'min_samples_overall': self.min_samples_overall,
                'min_samples_code': self.min_samples_code,
                'min_samples_month': self.min_samples_month,
                'min_samples_code_month': self.min_samples_code_month
            },
            'data_summary': {
                'n_individual_models': len(self.loaded_predictions),
                'n_family_ensembles': len(self.family_ensembles),
                'n_global_ensembles': len(self.global_ensembles),
                'total_models_evaluated': len(self.all_predictions),
                'n_evaluation_records': len(self.evaluation_results)
            },
            'validation_summary': {
                'total_warnings': sum(len(v.get('warnings', [])) for v in self.validation_results.values()),
                'models_with_warnings': sum(1 for v in self.validation_results.values() if v.get('warnings'))
            }
        }
        
        # Add model lists
        self.metadata['models'] = {
            'individual_models': list(self.loaded_predictions.keys()),
            'family_ensembles': list(self.family_ensembles.keys()),
            'global_ensembles': list(self.global_ensembles.keys())
        }
    
    def run_pipeline(self) -> bool:
        """
        Run the complete evaluation pipeline.
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        logger.info("Starting evaluation pipeline")
        start_time = datetime.now()
        
        try:
            # Step 1: Load predictions
            if not self.load_predictions():
                logger.error("Failed to load predictions")
                return False
            
            # Step 2: Create ensembles
            if not self.create_ensembles():
                logger.error("Failed to create ensembles")
                return False
            
            # Step 3: Evaluate models
            if not self.evaluate_models():
                logger.error("Failed to evaluate models")
                return False
            
            # Step 4: Generate outputs
            if not self.generate_outputs():
                logger.error("Failed to generate outputs")
                return False
            
            # Success
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
            logger.info(f"Total duration: {duration}")
            logger.info(f"Individual models: {len(self.loaded_predictions)}")
            logger.info(f"Family ensembles: {len(self.family_ensembles)}")
            logger.info(f"Global ensembles: {len(self.global_ensembles)}")
            logger.info(f"Evaluation records: {len(self.evaluation_results)}")
            logger.info(f"Output directory: {self.output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            return False

def run_evaluation_pipeline(results_dir: str = "../monthly_forecasting_results",
                           output_dir: str = "../monthly_forecasting_results/evaluation",
                           evaluation_day: Union[str, int] = 'end',
                           common_codes_only: bool = True,
                           ensemble_method: str = 'mean',
                           include_code_month: bool = False,
                           min_samples_overall: int = 10,
                           min_samples_code: int = 5,
                           min_samples_month: int = 3,
                           min_samples_code_month: int = 2) -> bool:
    """
    Convenience function to run the evaluation pipeline.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing model results
    output_dir : str
        Directory to save evaluation outputs
    evaluation_day : Union[str, int]
        Day of month for evaluation ('end' or integer)
    common_codes_only : bool
        Whether to use only common basin codes
    ensemble_method : str
        Method for ensemble creation
    include_code_month : bool
        Whether to include per-code-month evaluation
    min_samples_overall : int
        Minimum samples for overall evaluation
    min_samples_code : int
        Minimum samples for per-code evaluation
    min_samples_month : int
        Minimum samples for per-month evaluation
    min_samples_code_month : int
        Minimum samples for per-code-month evaluation
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    pipeline = EvaluationPipeline(
        results_dir=results_dir,
        output_dir=output_dir,
        evaluation_day=evaluation_day,
        common_codes_only=common_codes_only,
        ensemble_method=ensemble_method,
        include_code_month=include_code_month,
        min_samples_overall=min_samples_overall,
        min_samples_code=min_samples_code,
        min_samples_month=min_samples_month,
        min_samples_code_month=min_samples_code_month
    )
    
    return pipeline.run_pipeline()

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(description="Monthly Discharge Forecasting Evaluation Pipeline")
    
    parser.add_argument("--results_dir", type=str, 
                       default="../monthly_forecasting_results",
                       help="Directory containing model results")
    
    parser.add_argument("--output_dir", type=str,
                       default="../monthly_forecasting_results/evaluation",
                       help="Directory to save evaluation outputs")
    
    parser.add_argument("--evaluation_day", type=str, default="end",
                       help="Day of month for evaluation ('end' or integer)")
    
    parser.add_argument("--no_common_codes", action="store_true",
                       help="Don't filter to common basin codes")
    
    parser.add_argument("--ensemble_method", type=str, default="mean",
                       choices=["mean", "median", "weighted_mean"],
                       help="Method for ensemble creation")
    
    parser.add_argument("--include_code_month", action="store_true",
                       help="Include per-code-month evaluation")
    
    parser.add_argument("--min_samples_overall", type=int, default=10,
                       help="Minimum samples for overall evaluation")
    
    parser.add_argument("--min_samples_code", type=int, default=5,
                       help="Minimum samples for per-code evaluation")
    
    parser.add_argument("--min_samples_month", type=int, default=3,
                       help="Minimum samples for per-month evaluation")
    
    parser.add_argument("--min_samples_code_month", type=int, default=2,
                       help="Minimum samples for per-code-month evaluation")
    
    args = parser.parse_args()
    
    # Convert evaluation_day to int if it's a number
    evaluation_day = args.evaluation_day
    if evaluation_day != 'end':
        try:
            evaluation_day = int(evaluation_day)
        except ValueError:
            logger.error(f"Invalid evaluation_day: {evaluation_day}")
            return False
    
    # Run pipeline
    success = run_evaluation_pipeline(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        evaluation_day=evaluation_day,
        common_codes_only=not args.no_common_codes,
        ensemble_method=args.ensemble_method,
        include_code_month=args.include_code_month,
        min_samples_overall=args.min_samples_overall,
        min_samples_code=args.min_samples_code,
        min_samples_month=args.min_samples_month,
        min_samples_code_month=args.min_samples_code_month
    )
    
    if success:
        logger.info("Pipeline completed successfully")
        return True
    else:
        logger.error("Pipeline failed")
        return False

if __name__ == "__main__":
    main()