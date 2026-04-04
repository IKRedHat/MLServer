import os
import sys
import subprocess
import tempfile
import pytest


def test_mlflow_train_script_e2e():
    """
    End-to-end test for the MLflow training example script.
    Verifies that the script can be executed, downloads the dataset,
    trains the model, and outputs the expected metrics.
    """
    # Determine the path to the train.py script
    # Assuming the test is run from the repository root
    script_path = os.path.join("docs", "examples", "mlflow", "src", "train.py")
    
    if not os.path.exists(script_path):
        pytest.skip(f"Training script not found at {script_path}. Ensure tests are run from the repository root.")

    # Execute the script with custom alpha and l1_ratio arguments
    alpha = "0.3"
    l1_ratio = "0.7"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        # Set MLflow tracking URI to a local temp directory to avoid creating mlruns in the repo
        env["MLFLOW_TRACKING_URI"] = f"file://{tmpdir}"
        
        result = subprocess.run(
            [sys.executable, script_path, alpha, l1_ratio],
            capture_output=True,
            text=True,
            env=env
        )
        
        # Check that the script executed successfully
        assert result.returncode == 0, f"Script execution failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        
        # Verify the output contains the expected metrics
        expected_header = f"Elasticnet model (alpha={float(alpha):.6f}, l1_ratio={float(l1_ratio):.6f}):"
        assert expected_header in result.stdout, f"Expected header '{expected_header}' not found in stdout."
        assert "RMSE:" in result.stdout, "RMSE metric not found in stdout."
        assert "MAE:" in result.stdout, "MAE metric not found in stdout."
        assert "R2:" in result.stdout, "R2 metric not found in stdout."
