import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, call
import sys
import os
from io import StringIO

# Add the src directory to the path to import the train module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
from train import eval_metrics


@pytest.fixture
def mock_wine_data():
    """Create mock wine quality dataset that mimics the real structure."""
    np.random.seed(42)
    n_samples = 100
    
    # Create realistic wine quality features
    data = {
        'fixed acidity': np.random.uniform(4.6, 15.9, n_samples),
        'volatile acidity': np.random.uniform(0.12, 1.58, n_samples),
        'citric acid': np.random.uniform(0.0, 1.0, n_samples),
        'residual sugar': np.random.uniform(0.9, 15.5, n_samples),
        'chlorides': np.random.uniform(0.012, 0.611, n_samples),
        'free sulfur dioxide': np.random.uniform(1.0, 72.0, n_samples),
        'total sulfur dioxide': np.random.uniform(6.0, 289.0, n_samples),
        'density': np.random.uniform(0.99007, 1.00369, n_samples),
        'pH': np.random.uniform(2.74, 4.01, n_samples),
        'sulphates': np.random.uniform(0.33, 2.0, n_samples),
        'alcohol': np.random.uniform(8.4, 14.9, n_samples),
        'quality': np.random.randint(3, 10, n_samples)  # Wine quality scores 3-9
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_csv_content():
    """Create CSV content string that mimics the wine quality dataset."""
    return """fixed acidity;volatile acidity;citric acid;residual sugar;chlorides;free sulfur dioxide;total sulfur dioxide;density;pH;sulphates;alcohol;quality
7.4;0.7;0.0;1.9;0.076;11.0;34.0;0.9978;3.51;0.56;9.4;5
7.8;0.88;0.0;2.6;0.098;25.0;67.0;0.9968;3.2;0.68;9.8;5
7.8;0.76;0.04;2.3;0.092;15.0;54.0;0.997;3.26;0.65;9.8;5
11.2;0.28;0.56;1.9;0.075;17.0;60.0;0.998;3.16;0.58;9.8;6
7.4;0.7;0.0;1.9;0.076;11.0;34.0;0.9978;3.51;0.56;9.4;5"""


class TestEvalMetrics:
    """Test the eval_metrics function with known mathematical results."""
    
    def test_eval_metrics_perfect_prediction(self):
        """Test eval_metrics with perfect predictions (all metrics should be optimal)."""
        actual = np.array([5, 6, 7, 8, 5])
        pred = np.array([5, 6, 7, 8, 5])
        
        rmse, mae, r2 = eval_metrics(actual, pred)
        
        assert rmse == 0.0, f"RMSE should be 0 for perfect prediction, got {rmse}"
        assert mae == 0.0, f"MAE should be 0 for perfect prediction, got {mae}"
        assert r2 == 1.0, f"R2 should be 1 for perfect prediction, got {r2}"
    
    def test_eval_metrics_known_values(self):
        """Test eval_metrics with known input/output pairs."""
        actual = np.array([3, 5, 7, 9])
        pred = np.array([2, 6, 8, 10])
        
        rmse, mae, r2 = eval_metrics(actual, pred)
        
        # Expected values calculated manually:
        # MSE = ((3-2)^2 + (5-6)^2 + (7-8)^2 + (9-10)^2) / 4 = (1+1+1+1)/4 = 1
        # RMSE = sqrt(1) = 1.0
        # MAE = (|3-2| + |5-6| + |7-8| + |9-10|) / 4 = (1+1+1+1)/4 = 1.0
        
        assert abs(rmse - 1.0) < 1e-10, f"Expected RMSE=1.0, got {rmse}"
        assert abs(mae - 1.0) < 1e-10, f"Expected MAE=1.0, got {mae}"
        assert r2 > 0.8, f"R2 should be high for this prediction, got {r2}"
    
    def test_eval_metrics_with_dataframe_input(self):
        """Test eval_metrics works with pandas DataFrame input (as used in main script)."""
        actual_df = pd.DataFrame({'quality': [5, 6, 7, 8]})
        pred_array = np.array([5.1, 5.9, 7.1, 7.9])
        
        rmse, mae, r2 = eval_metrics(actual_df, pred_array)
        
        assert isinstance(rmse, float), "RMSE should be a float"
        assert isinstance(mae, float), "MAE should be a float"
        assert isinstance(r2, float), "R2 should be a float"
        assert rmse > 0, "RMSE should be positive"
        assert mae > 0, "MAE should be positive"


class TestTrainingPipelineE2E:
    """End-to-end tests for the complete training pipeline."""
    
    @patch('train.pd.read_csv')
    @patch('train.mlflow.start_run')
    @patch('train.mlflow.log_param')
    @patch('train.mlflow.log_metric')
    @patch('train.mlflow.sklearn.log_model')
    @patch('train.mlflow.get_tracking_uri')
    def test_complete_training_pipeline_file_store(self, mock_get_tracking_uri, mock_log_model, 
                                                  mock_log_metric, mock_log_param, mock_start_run, 
                                                  mock_read_csv, mock_wine_data):
        """Test the complete training pipeline with file store tracking URI."""
        # Setup mocks
        mock_read_csv.return_value = mock_wine_data
        mock_get_tracking_uri.return_value = "file:///tmp/mlruns"
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()
        
        # Capture stdout to verify print statements
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output), \
             patch('sys.argv', ['train.py', '0.3', '0.7']):
            
            # Import and execute the main script logic
            import train
            exec(open(os.path.join(os.path.dirname(train.__file__), 'train.py')).read())
        
        # Verify MLflow logging calls
        mock_log_param.assert_any_call("alpha", 0.3)
        mock_log_param.assert_any_call("l1_ratio", 0.7)
        
        # Verify metrics were logged (exact values will vary due to randomness)
        metric_calls = [call[0][0] for call in mock_log_metric.call_args_list]
        assert "rmse" in metric_calls
        assert "r2" in metric_calls
        assert "mae" in metric_calls
        
        # Verify model was logged without registration (file store)
        mock_log_model.assert_called_once()
        call_args = mock_log_model.call_args
        assert "model" in call_args[0] or "model" in call_args[1]
        # For file store, should not have registered_model_name
        if 'registered_model_name' in str(call_args):
            pytest.fail("Model should not be registered with file store")
        
        # Verify output contains expected information
        output = captured_output.getvalue()
        assert "Elasticnet model" in output
        assert "RMSE:" in output
        assert "MAE:" in output
        assert "R2:" in output
    
    @patch('train.pd.read_csv')
    @patch('train.mlflow.start_run')
    @patch('train.mlflow.log_param')
    @patch('train.mlflow.log_metric')
    @patch('train.mlflow.sklearn.log_model')
    @patch('train.mlflow.get_tracking_uri')
    def test_complete_training_pipeline_remote_store(self, mock_get_tracking_uri, mock_log_model, 
                                                    mock_log_metric, mock_log_param, mock_start_run, 
                                                    mock_read_csv, mock_wine_data):
        """Test the complete training pipeline with remote tracking URI (model registration)."""
        # Setup mocks
        mock_read_csv.return_value = mock_wine_data
        mock_get_tracking_uri.return_value = "http://localhost:5000"
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()
        
        with patch('sys.argv', ['train.py']):  # Use default parameters
            # Import and execute the main script logic
            import train
            exec(open(os.path.join(os.path.dirname(train.__file__), 'train.py')).read())
        
        # Verify default parameters were used
        mock_log_param.assert_any_call("alpha", 0.5)
        mock_log_param.assert_any_call("l1_ratio", 0.5)
        
        # Verify model was logged with registration (non-file store)
        mock_log_model.assert_called_once()
        call_args = mock_log_model.call_args
        # For remote store, should have registered_model_name
        assert 'registered_model_name' in str(call_args), "Model should be registered with remote store"
        assert 'ElasticnetWineModel' in str(call_args), "Should use correct model name"


class TestCommandLineArguments:
    """Test command-line argument handling."""
    
    @patch('train.pd.read_csv')
    @patch('train.mlflow.start_run')
    @patch('train.mlflow.log_param')
    @patch('train.mlflow.log_metric')
    @patch('train.mlflow.sklearn.log_model')
    @patch('train.mlflow.get_tracking_uri')
    def test_custom_alpha_l1_ratio(self, mock_get_tracking_uri, mock_log_model, 
                                  mock_log_metric, mock_log_param, mock_start_run, 
                                  mock_read_csv, mock_wine_data):
        """Test that custom alpha and l1_ratio parameters are handled correctly."""
        # Setup mocks
        mock_read_csv.return_value = mock_wine_data
        mock_get_tracking_uri.return_value = "file:///tmp/mlruns"
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()
        
        with patch('sys.argv', ['train.py', '0.1', '0.9']):
            # Import and execute the main script logic
            import train
            exec(open(os.path.join(os.path.dirname(train.__file__), 'train.py')).read())
        
        # Verify custom parameters were logged
        mock_log_param.assert_any_call("alpha", 0.1)
        mock_log_param.assert_any_call("l1_ratio", 0.9)
    
    @patch('train.pd.read_csv')
    @patch('train.mlflow.start_run')
    @patch('train.mlflow.log_param')
    @patch('train.mlflow.log_metric')
    @patch('train.mlflow.sklearn.log_model')
    @patch('train.mlflow.get_tracking_uri')
    def test_partial_arguments(self, mock_get_tracking_uri, mock_log_model, 
                              mock_log_metric, mock_log_param, mock_start_run, 
                              mock_read_csv, mock_wine_data):
        """Test that providing only alpha uses default l1_ratio."""
        # Setup mocks
        mock_read_csv.return_value = mock_wine_data
        mock_get_tracking_uri.return_value = "file:///tmp/mlruns"
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()
        
        with patch('sys.argv', ['train.py', '0.2']):
            # Import and execute the main script logic
            import train
            exec(open(os.path.join(os.path.dirname(train.__file__), 'train.py')).read())
        
        # Verify custom alpha and default l1_ratio were logged
        mock_log_param.assert_any_call("alpha", 0.2)
        mock_log_param.assert_any_call("l1_ratio", 0.5)  # default value


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @patch('train.pd.read_csv')
    @patch('train.logger.exception')
    def test_csv_download_failure(self, mock_logger_exception, mock_read_csv):
        """Test that CSV download failures are handled gracefully."""
        # Setup mock to raise exception
        mock_read_csv.side_effect = Exception("Network error")
        
        with patch('sys.argv', ['train.py']):
            try:
                # Import and execute the main script logic
                import train
                exec(open(os.path.join(os.path.dirname(train.__file__), 'train.py')).read())
            except Exception:
                pass  # Expected to fail after logging
        
        # Verify error was logged
        mock_logger_exception.assert_called_once()
        call_args = mock_logger_exception.call_args[0]
        assert "Unable to download training & test CSV" in call_args[0]


class TestModelRegistrationLogic:
    """Test the conditional model registration logic."""
    
    @pytest.mark.parametrize("tracking_uri,should_register", [
        ("file:///tmp/mlruns", False),
        ("file://./mlruns", False),
        ("http://localhost:5000", True),
        ("https://mlflow.example.com", True),
        ("databricks", True),
        ("sqlite:///mlflow.db", True),
    ])
    @patch('train.pd.read_csv')
    @patch('train.mlflow.start_run')
    @patch('train.mlflow.log_param')
    @patch('train.mlflow.log_metric')
    @patch('train.mlflow.sklearn.log_model')
    @patch('train.mlflow.get_tracking_uri')
    def test_model_registration_scenarios(self, mock_get_tracking_uri, mock_log_model, 
                                        mock_log_metric, mock_log_param, mock_start_run, 
                                        mock_read_csv, mock_wine_data, tracking_uri, should_register):
        """Test model registration behavior for different tracking URI types."""
        # Setup mocks
        mock_read_csv.return_value = mock_wine_data
        mock_get_tracking_uri.return_value = tracking_uri
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()
        
        with patch('sys.argv', ['train.py']):
            # Import and execute the main script logic
            import train
            exec(open(os.path.join(os.path.dirname(train.__file__), 'train.py')).read())
        
        # Verify model logging behavior
        mock_log_model.assert_called_once()
        call_args = mock_log_model.call_args
        
        if should_register:
            assert 'registered_model_name' in str(call_args), f"Model should be registered for {tracking_uri}"
            assert 'ElasticnetWineModel' in str(call_args), "Should use correct model name"
        else:
            assert 'registered_model_name' not in str(call_args), f"Model should not be registered for {tracking_uri}"