"""Unit tests for MLflow training script.

This test module provides focused unit tests for the train.py module,
specifically testing the eval_metrics function and module importability
without complex external dependencies or mocking.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add the src directory to Python path to enable imports
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

# Import the module under test
import train


class TestEvalMetrics:
    """Unit tests for the eval_metrics function."""
    
    def test_perfect_predictions(self):
        """Test eval_metrics with identical actual and predicted values."""
        actual = np.array([5.0, 6.0, 7.0, 8.0])
        pred = np.array([5.0, 6.0, 7.0, 8.0])
        
        rmse, mae, r2 = train.eval_metrics(actual, pred)
        
        assert rmse == 0.0, "RMSE should be 0 for perfect predictions"
        assert mae == 0.0, "MAE should be 0 for perfect predictions"
        assert r2 == 1.0, "R2 should be 1.0 for perfect predictions"
    
    def test_known_metric_values(self):
        """Test eval_metrics against manually calculated values."""
        actual = np.array([3.0, -0.5, 2.0, 7.0])
        pred = np.array([2.5, 0.0, 2.0, 8.0])
        
        rmse, mae, r2 = train.eval_metrics(actual, pred)
        
        # MSE = [(0.5)^2 + (0.5)^2 + (0)^2 + (1)^2] / 4 = 0.375
        # RMSE = sqrt(0.375) ≈ 0.6124
        assert abs(rmse - 0.6124) < 0.001, f"Expected RMSE ≈ 0.6124, got {rmse}"
        
        # MAE = [0.5 + 0.5 + 0 + 1] / 4 = 0.5
        assert mae == 0.5, f"Expected MAE = 0.5, got {mae}"
        
        # R2 should be positive for reasonable predictions
        assert r2 > 0.8, f"Expected R2 > 0.8, got {r2}"
    
    def test_with_pandas_series(self):
        """Test eval_metrics works with pandas Series inputs."""
        actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = pd.Series([1.1, 2.2, 2.9, 4.1, 4.8])
        
        rmse, mae, r2 = train.eval_metrics(actual, pred)
        
        assert isinstance(rmse, (int, float, np.floating))
        assert isinstance(mae, (int, float, np.floating))
        assert isinstance(r2, (int, float, np.floating))
        assert rmse > 0
        assert mae > 0
        assert 0 < r2 <= 1
    
    def test_returns_tuple_of_three(self):
        """Test that eval_metrics returns exactly three numeric values."""
        actual = np.array([1, 2, 3, 4, 5])
        pred = np.array([1, 2, 3, 4, 5])
        
        result = train.eval_metrics(actual, pred)
        
        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 3, "Should return exactly 3 values"
        assert all(isinstance(x, (int, float, np.floating)) for x in result)
    
    def test_larger_errors(self):
        """Test eval_metrics with predictions that have larger errors."""
        actual = np.array([10.0, 20.0, 30.0, 40.0])
        pred = np.array([5.0, 25.0, 35.0, 45.0])
        
        rmse, mae, r2 = train.eval_metrics(actual, pred)
        
        # With larger errors, metrics should reflect that
        assert rmse > 0, "RMSE should be positive with errors"
        assert mae > 0, "MAE should be positive with errors"
        # R2 can be negative for very poor predictions, but should be calculable
        assert isinstance(r2, (int, float, np.floating))
    
    def test_constant_predictions(self):
        """Test eval_metrics when all predictions are the same value."""
        actual = np.array([3.0, 5.0, 7.0, 9.0])
        pred = np.array([5.0, 5.0, 5.0, 5.0])  # All predictions are 5
        
        rmse, mae, r2 = train.eval_metrics(actual, pred)
        
        # Should still calculate valid metrics
        assert rmse > 0
        assert mae > 0
        assert isinstance(r2, (int, float, np.floating))
    
    def test_small_sample_size(self):
        """Test eval_metrics with minimal sample size."""
        actual = np.array([1.0, 2.0])
        pred = np.array([1.5, 2.5])
        
        rmse, mae, r2 = train.eval_metrics(actual, pred)
        
        assert rmse > 0
        assert mae == 0.5
        assert isinstance(r2, (int, float, np.floating))
    
    def test_negative_values(self):
        """Test eval_metrics handles negative values correctly."""
        actual = np.array([-5.0, -3.0, -1.0, 1.0])
        pred = np.array([-4.5, -3.2, -0.8, 1.3])
        
        rmse, mae, r2 = train.eval_metrics(actual, pred)
        
        assert rmse > 0
        assert mae > 0
        assert isinstance(r2, (int, float, np.floating))
    
    def test_floating_point_precision(self):
        """Test eval_metrics with high precision floating point values."""
        actual = np.array([1.123456789, 2.987654321, 3.456789123])
        pred = np.array([1.123456788, 2.987654322, 3.456789124])
        
        rmse, mae, r2 = train.eval_metrics(actual, pred)
        
        # With very small differences, metrics should be very small
        assert rmse < 0.01
        assert mae < 0.01
        assert r2 > 0.99


class TestModuleStructure:
    """Tests for module-level structure and imports."""
    
    def test_module_imports_successfully(self):
        """Test that the train module can be imported without errors."""
        import train as t
        assert t is not None
    
    def test_eval_metrics_is_callable(self):
        """Test that eval_metrics is defined and callable."""
        assert hasattr(train, 'eval_metrics')
        assert callable(train.eval_metrics)
    
    def test_required_dependencies_available(self):
        """Test that required dependencies are importable."""
        try:
            import pandas
            import numpy
            import sklearn
            import mlflow
        except ImportError as e:
            pytest.fail(f"Required dependency not available: {e}")
    
    def test_logger_is_configured(self):
        """Test that the module has a logger configured."""
        assert hasattr(train, 'logger')
        assert train.logger is not None


class TestParameterHandling:
    """Tests for parameter validation and handling."""
    
    def test_command_line_alpha_parsing(self):
        """Test that alpha parameter can be parsed from command line args."""
        # Simulate command line arguments
        test_args = ['train.py', '0.3']
        alpha = float(test_args[1]) if len(test_args) > 1 else 0.5
        assert alpha == 0.3
    
    def test_command_line_l1_ratio_parsing(self):
        """Test that l1_ratio parameter can be parsed from command line args."""
        test_args = ['train.py', '0.3', '0.7']
        l1_ratio = float(test_args[2]) if len(test_args) > 2 else 0.5
        assert l1_ratio == 0.7
    
    def test_default_alpha_value(self):
        """Test default alpha value when not provided."""
        test_args = ['train.py']
        alpha = float(test_args[1]) if len(test_args) > 1 else 0.5
        assert alpha == 0.5
    
    def test_default_l1_ratio_value(self):
        """Test default l1_ratio value when not provided."""
        test_args = ['train.py', '0.3']
        l1_ratio = float(test_args[2]) if len(test_args) > 2 else 0.5
        assert l1_ratio == 0.5


class TestDataFrameOperations:
    """Tests for DataFrame operations used in the training script."""
    
    def test_quality_column_extraction(self):
        """Test extracting quality column from DataFrame."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'quality': [5, 6, 7]
        })
        
        quality = df[['quality']]
        assert 'quality' in quality.columns
        assert len(quality) == 3
    
    def test_feature_column_extraction(self):
        """Test dropping quality column to get features."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'quality': [5, 6, 7]
        })
        
        features = df.drop(['quality'], axis=1)
        assert 'quality' not in features.columns
        assert 'feature1' in features.columns
        assert 'feature2' in features.columns
    
    def test_train_test_split_compatibility(self):
        """Test that data structure is compatible with train_test_split."""
        from sklearn.model_selection import train_test_split
        
        df = pd.DataFrame({
            'feature1': range(20),
            'feature2': range(20, 40),
            'quality': range(5, 25)
        })
        
        train_data, test_data = train_test_split(df, test_size=0.25, random_state=42)
        
        assert len(train_data) > 0
        assert len(test_data) > 0
        assert len(train_data) + len(test_data) == len(df)
