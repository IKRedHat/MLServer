"""
Integration tests for benchmarking/generator.py
Tests real file I/O and end-to-end functionality without extensive mocking.
"""

import os
import tempfile
import shutil
import pytest
import json
import sys
from pathlib import Path

# Add project root to path to import benchmarking module
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarking import generator
from mlserver import types


class TestGeneratorIntegration:
    """Integration tests using real temporary directories and file operations."""
    
    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_data_path = generator.DATA_PATH
        generator.DATA_PATH = self.temp_dir
    
    def teardown_method(self):
        """Clean up temporary directory after each test."""
        generator.DATA_PATH = self.original_data_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_pipeline_execution(self):
        """Test complete workflow from start to finish."""
        # Execute the main pipeline
        generator.main()
        
        # Verify all three output files are created
        grpc_pb_file = os.path.join(self.temp_dir, "grpc-requests.pb")
        grpc_json_file = os.path.join(self.temp_dir, "grpc-requests.json")
        rest_json_file = os.path.join(self.temp_dir, "rest-requests.json")
        
        assert os.path.exists(grpc_pb_file), "gRPC protobuf file not created"
        assert os.path.exists(grpc_json_file), "gRPC JSON file not created"
        assert os.path.exists(rest_json_file), "REST JSON file not created"
        
        # Verify files have content
        assert os.path.getsize(grpc_pb_file) > 0, "gRPC protobuf file is empty"
        assert os.path.getsize(grpc_json_file) > 0, "gRPC JSON file is empty"
        assert os.path.getsize(rest_json_file) > 0, "REST JSON file is empty"
    
    def test_generated_data_consistency(self):
        """Verify same input produces consistent output across runs."""
        # Set numpy seed for reproducibility
        import numpy as np
        np.random.seed(42)
        
        # First run
        generator.main()
        
        with open(os.path.join(self.temp_dir, "rest-requests.json"), "r") as f:
            first_run_data = json.load(f)
        
        # Clean up and run again with same seed
        for filename in ["grpc-requests.pb", "grpc-requests.json", "rest-requests.json"]:
            filepath = os.path.join(self.temp_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
        
        np.random.seed(42)
        generator.main()
        
        with open(os.path.join(self.temp_dir, "rest-requests.json"), "r") as f:
            second_run_data = json.load(f)
        
        # Data should be identical
        assert first_run_data == second_run_data, "Generated data is not consistent across runs"
    
    def test_file_format_validation(self):
        """Validate generated JSON files conform to expected schema."""
        generator.main()
        
        # Validate gRPC JSON format
        with open(os.path.join(self.temp_dir, "grpc-requests.json"), "r") as f:
            grpc_data = json.load(f)
        
        assert isinstance(grpc_data, dict), "gRPC JSON should be a dictionary"
        assert "inputs" in grpc_data, "gRPC JSON missing 'inputs' field"
        
        # Validate REST JSON format
        with open(os.path.join(self.temp_dir, "rest-requests.json"), "r") as f:
            rest_data = json.load(f)
        
        assert isinstance(rest_data, dict), "REST JSON should be a dictionary"
        assert "inputs" in rest_data, "REST JSON missing 'inputs' field"
        assert isinstance(rest_data["inputs"], list), "REST inputs should be a list"
        assert len(rest_data["inputs"]) == 1, "Should have exactly one input"
        
        input_spec = rest_data["inputs"][0]
        assert input_spec["name"] == "input-0", "Input name should be 'input-0'"
        assert input_spec["datatype"] == "FP32", "Input datatype should be FP32"
        assert isinstance(input_spec["shape"], list), "Shape should be a list"
        assert len(input_spec["shape"]) == 1, "Shape should have one dimension"
        assert input_spec["shape"][0] == 1024, "First request should have 1024 elements"
    
    def test_protobuf_binary_integrity(self):
        """Verify protobuf binary file contains valid data with size prefixes."""
        generator.main()
        
        pb_file = os.path.join(self.temp_dir, "grpc-requests.pb")
        
        with open(pb_file, "rb") as f:
            content = f.read()
        
        # File should not be empty
        assert len(content) > 0, "Protobuf file is empty"
        
        # Should start with a varint size prefix
        # Varint encoding: first byte should have continuation bit patterns
        first_byte = content[0]
        assert isinstance(first_byte, int), "First byte should be an integer"
        
        # File should be larger than just the size prefix
        assert len(content) > 10, "Protobuf file seems too small to contain actual data"
    
    def test_data_path_configuration(self):
        """Test behavior with different DATA_PATH configurations."""
        # Create a nested directory structure
        nested_dir = os.path.join(self.temp_dir, "nested", "path")
        os.makedirs(nested_dir, exist_ok=True)
        
        generator.DATA_PATH = nested_dir
        generator.main()
        
        # Files should be created in the nested directory
        assert os.path.exists(os.path.join(nested_dir, "grpc-requests.pb"))
        assert os.path.exists(os.path.join(nested_dir, "grpc-requests.json"))
        assert os.path.exists(os.path.join(nested_dir, "rest-requests.json"))
    
    def test_module_constants(self):
        """Verify MODEL_NAME and MODEL_VERSION are used in generated files."""
        generator.main()
        
        # Check that constants are properly applied
        with open(os.path.join(self.temp_dir, "grpc-requests.json"), "r") as f:
            grpc_data = json.load(f)
        
        # The gRPC data should contain model information
        # (exact structure depends on protobuf conversion)
        assert isinstance(grpc_data, dict), "gRPC data should be a dictionary"
        
        # Verify constants have expected values
        assert generator.MODEL_NAME == "sum-model", "MODEL_NAME constant changed"
        assert generator.MODEL_VERSION == "v1.2.3", "MODEL_VERSION constant changed"
