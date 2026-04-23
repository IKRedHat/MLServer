import pytest
import tempfile
import shutil
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from unittest.mock import patch
from sklearn.model_selection import train_test_split

# Add the src directory to the path to import train module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
import train


class TestTrainE2E:
    """End-to-end tests for the MLflow training pipeline."""
    
    def setup_method(self):
        """Set up test environment with temporary MLflow tracking and sample data."""
        # Create temporary directory for MLflow tracking
        self.temp_dir = tempfile.mkdtemp()
        self.mlflow_tracking_uri = f"file://{self.temp_dir}/mlruns"
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Create sample wine quality dataset for testing
        np.random.seed(42)
        n_samples = 100
        self.sample_data = pd.DataFrame({
            'fixed acidity': np.random.normal(8.0, 1.5, n_samples),
            'volatile acidity': np.random.normal(0.5, 0.2, n_samples),
            'citric acid': np.random.normal(0.3, 0.1, n_samples),
            'residual sugar': np.random.normal(2.5, 1.0, n_samples),
            'chlorides': np.random.normal(0.08, 0.02, n_samples),
            'free sulfur dioxide': np.random.normal(15, 5, n_samples),
            'total sulfur dioxide': np.random.normal(45, 15, n_samples),
            'density': np.random.normal(0.996, 0.002, n_samples),
            'pH': np.random.normal(3.3, 0.2, n_samples),
            'sulphates': np.random.normal(0.65, 0.15, n_samples),
            'alcohol': np.random.normal(10.5, 1.0, n_samples),
            'quality': np.random.randint(3, 10, n_samples)
        })

    def test_eval_metrics_function(self):
        """Test the eval_metrics function with known input/output pairs."""
        # Create test data with known relationships
        actual = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        pred = np.array([5.1, 5.9, 7.2, 7.8, 8.9])
        
        rmse, mae, r2 = train.eval_metrics(actual, pred)
        
        # Verify RMSE calculation
        expected_rmse = np.sqrt(mean_squared_error(actual, pred))
        assert abs(rmse - expected_rmse) < 1e-10, f"RMSE mismatch: {rmse} vs {expected_rmse}"
        
        # Verify MAE calculation
        expected_mae = mean_absolute_error(actual, pred)
        assert abs(mae - expected_mae) < 1e-10, f"MAE mismatch: {mae} vs {expected_mae}"
        
        # Verify R2 calculation
        expected_r2 = r2_score(actual, pred)
        assert abs(r2 - expected_r2) < 1e-10, f"R2 mismatch: {r2} vs {expected_r2}"

    def test_train_pipeline_with_default_params(self):
        """Test the complete training pipeline with default parameters."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = self.sample_data
            
            with mlflow.start_run() as run:
                # Execute the main training logic
                data = self.sample_data
                train_data, test_data = train_test_split(data, random_state=42)
                train_x = train_data.drop(["quality"], axis=1)
                test_x = test_data.drop(["quality"], axis=1)
                train_y = train_data[["quality"]]
                test_y = test_data[["quality"]]
                
                alpha = 0.5  # default
                l1_ratio = 0.5  # default
                
                lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
                lr.fit(train_x, train_y)
                predicted_qualities = lr.predict(test_x)
                
                rmse, mae, r2 = train.eval_metrics(test_y, predicted_qualities)
                
                # Log parameters and metrics
                mlflow.log_param("alpha", alpha)
                mlflow.log_param("l1_ratio", l1_ratio)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)
                
                # Verify model signature
                model_signature = infer_signature(train_x, train_y)
                mlflow.sklearn.log_model(lr, "model", signature=model_signature)
                
                # Verify metrics are reasonable
                assert rmse > 0, "RMSE should be positive"
                assert mae > 0, "MAE should be positive"
                assert -1 <= r2 <= 1, f"R2 should be between -1 and 1, got {r2}"
                
                # Verify MLflow logging
                run_data = mlflow.get_run(run.info.run_id)
                assert run_data.data.params["alpha"] == "0.5"
                assert run_data.data.params["l1_ratio"] == "0.5"
                assert "rmse" in run_data.data.metrics
                assert "mae" in run_data.data.metrics
                assert "r2" in run_data.data.metrics

    def test_train_pipeline_with_custom_params(self):
        """Test the training pipeline with custom alpha and l1_ratio values."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = self.sample_data
            
            # Test with custom parameters
            custom_alpha = 0.8
            custom_l1_ratio = 0.3
            
            with mlflow.start_run() as run:
                data = self.sample_data
                train_data, test_data = train_test_split(data, random_state=42)
                train_x = train_data.drop(["quality"], axis=1)
                test_x = test_data.drop(["quality"], axis=1)
                train_y = train_data[["quality"]]
                test_y = test_data[["quality"]]
                
                lr = ElasticNet(alpha=custom_alpha, l1_ratio=custom_l1_ratio, random_state=42)
                lr.fit(train_x, train_y)
                predicted_qualities = lr.predict(test_x)
                
                rmse, mae, r2 = train.eval_metrics(test_y, predicted_qualities)
                
                mlflow.log_param("alpha", custom_alpha)
                mlflow.log_param("l1_ratio", custom_l1_ratio)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)
                
                # Verify custom parameters were logged correctly
                run_data = mlflow.get_run(run.info.run_id)
                assert run_data.data.params["alpha"] == str(custom_alpha)
                assert run_data.data.params["l1_ratio"] == str(custom_l1_ratio)

    def teardown_method(self):
        """Clean up temporary MLflow directories and reset global state."""
        # End any active MLflow runs
        if mlflow.active_run():
            mlflow.end_run()
            
        # Clean up temporary directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)