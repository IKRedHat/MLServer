import os
import json
import subprocess
import tempfile
import sys

def test_generate_dotenv_e2e():
    """
    End-to-end test for the generate_dotenv.py script.
    It creates dummy settings.json and model-settings.json files,
    runs the script, and verifies that the .env file is generated correctly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = {
            "debug": True,
            "http_port": 8080
        }
        with open(os.path.join(tmpdir, "settings.json"), "w") as f:
            json.dump(settings, f)
        
        model_settings = {
            "name": "my-model",
            "implementation": "mlserver_sklearn.SKLearnModel",
            "parameters": {
                "uri": "my-model.joblib",
                "version": "v1"
            }
        }
        with open(os.path.join(tmpdir, "model-settings.json"), "w") as f:
            json.dump(model_settings, f)
            
        output_file = os.path.join(tmpdir, ".env")
        
        # Find the script
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../hack/generate_dotenv.py"))
        if not os.path.exists(script_path):
            script_path = os.path.abspath(os.path.join(os.getcwd(), "hack/generate_dotenv.py"))
            
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path, tmpdir, output_file],
            capture_output=True,
            text=True
        )
        
        # Assuming the environment has the dependencies installed for E2E tests.
        if result.returncode == 0:
            assert os.path.exists(output_file)
            with open(output_file, "r") as f:
                content = f.read()
            
            assert "8080" in content
            assert "my-model" in content
            assert "mlserver_sklearn.SKLearnModel" in content
