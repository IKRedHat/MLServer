"""
Tests for benchmarking/generator.py
"""

import os
import json
import tempfile
import shutil
import pytest
import numpy as np

from benchmarking import generator
from mlserver import types


class TestGenerateTestRequests:
    def test_returns_list_of_inference_requests(self):
        """Test that generate_test_requests returns a list of InferenceRequest objects."""
        requests = generator.generate_test_requests()
        
        assert isinstance(requests, list)
        assert len(requests) > 0
        assert all(isinstance(req, types.InferenceRequest) for req in requests)
    
    def test_generates_correct_number_of_requests(self):
        """Test that the correct number of requests are generated."""
        requests = generator.generate_test_requests()
        
        # Should generate requests for contents_lens from 2^10 to 2^15
        # That's 6 different sizes
        assert len(requests) == 6
    
    def test_request_structure(self):
        """Test that each request has the correct structure."""
        requests = generator.generate_test_requests()
        
        for req in requests:
            assert len(req.inputs) == 1
            assert req.inputs[0].name == "input-0"
            assert req.inputs[0].datatype == "FP32"
            assert len(req.inputs[0].shape) == 1
    
    def test_request_sizes(self):
        """Test that requests have the expected sizes."""
        requests = generator.generate_test_requests()
        
        expected_sizes = [2**10, 2**11, 2**12, 2**13, 2**14, 2**15]
        actual_sizes = [req.inputs[0].shape[0] for req in requests]
        
        assert actual_sizes == expected_sizes
    
    def test_request_data_range(self):
        """Test that generated data is within expected range."""
        requests = generator.generate_test_requests()
        
        for req in requests:
            data = req.inputs[0].data.__root__
            assert all(0 <= val <= 9999 for val in data)


class TestSaveGrpcRequests:
    @pytest.fixture
    def temp_data_path(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        original_data_path = generator.DATA_PATH
        generator.DATA_PATH = temp_dir
        yield temp_dir
        generator.DATA_PATH = original_data_path
        shutil.rmtree(temp_dir)
    
    def test_saves_grpc_pb_file(self, temp_data_path):
        """Test that save_grpc_requests creates a .pb file."""
        requests = generator.generate_test_requests()
        generator.save_grpc_requests(requests)
        
        pb_file = os.path.join(temp_data_path, "grpc-requests.pb")
        assert os.path.exists(pb_file)
        assert os.path.getsize(pb_file) > 0
    
    def test_saves_grpc_json_file(self, temp_data_path):
        """Test that save_grpc_requests creates a .json file."""
        requests = generator.generate_test_requests()
        generator.save_grpc_requests(requests)
        
        json_file = os.path.join(temp_data_path, "grpc-requests.json")
        assert os.path.exists(json_file)
        
        with open(json_file, "r") as f:
            data = json.load(f)
            assert isinstance(data, dict)
            assert "inputs" in data
    
    def test_uses_only_first_request(self, temp_data_path):
        """Test that only the first request is saved for gRPC."""
        requests = generator.generate_test_requests()
        generator.save_grpc_requests(requests)
        
        json_file = os.path.join(temp_data_path, "grpc-requests.json")
        with open(json_file, "r") as f:
            data = json.load(f)
            # Verify it contains model name and version
            assert "modelName" in data or "inputs" in data


class TestSaveRestRequests:
    @pytest.fixture
    def temp_data_path(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        original_data_path = generator.DATA_PATH
        generator.DATA_PATH = temp_dir
        yield temp_dir
        generator.DATA_PATH = original_data_path
        shutil.rmtree(temp_dir)
    
    def test_saves_rest_json_file(self, temp_data_path):
        """Test that save_rest_requests creates a JSON file."""
        requests = generator.generate_test_requests()
        generator.save_rest_requests(requests)
        
        json_file = os.path.join(temp_data_path, "rest-requests.json")
        assert os.path.exists(json_file)
        
        with open(json_file, "r") as f:
            data = json.load(f)
            assert isinstance(data, dict)
            assert "inputs" in data
    
    def test_uses_first_request(self, temp_data_path):
        """Test that the first request is saved for REST."""
        requests = generator.generate_test_requests()
        generator.save_rest_requests(requests)
        
        json_file = os.path.join(temp_data_path, "rest-requests.json")
        with open(json_file, "r") as f:
            data = json.load(f)
            assert len(data["inputs"]) == 1
            assert data["inputs"][0]["name"] == "input-0"
            assert data["inputs"][0]["datatype"] == "FP32"
    
    def test_saved_data_is_valid_json(self, temp_data_path):
        """Test that saved REST request is valid JSON with correct structure."""
        requests = generator.generate_test_requests()
        generator.save_rest_requests(requests)
        
        json_file = os.path.join(temp_data_path, "rest-requests.json")
        with open(json_file, "r") as f:
            data = json.load(f)
            assert data["inputs"][0]["shape"][0] == 1024  # First request has 2^10 elements


class TestMain:
    @pytest.fixture
    def temp_data_path(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        original_data_path = generator.DATA_PATH
        generator.DATA_PATH = temp_dir
        yield temp_dir
        generator.DATA_PATH = original_data_path
        shutil.rmtree(temp_dir)
    
    def test_main_creates_all_files(self, temp_data_path):
        """Test that main() creates all expected output files."""
        generator.main()
        
        grpc_pb_file = os.path.join(temp_data_path, "grpc-requests.pb")
        grpc_json_file = os.path.join(temp_data_path, "grpc-requests.json")
        rest_json_file = os.path.join(temp_data_path, "rest-requests.json")
        
        assert os.path.exists(grpc_pb_file)
        assert os.path.exists(grpc_json_file)
        assert os.path.exists(rest_json_file)
    
    def test_main_creates_valid_files(self, temp_data_path):
        """Test that main() creates valid, non-empty files."""
        generator.main()
        
        grpc_pb_file = os.path.join(temp_data_path, "grpc-requests.pb")
        grpc_json_file = os.path.join(temp_data_path, "grpc-requests.json")
        rest_json_file = os.path.join(temp_data_path, "rest-requests.json")
        
        # Check files are not empty
        assert os.path.getsize(grpc_pb_file) > 0
        assert os.path.getsize(grpc_json_file) > 0
        assert os.path.getsize(rest_json_file) > 0
        
        # Check JSON files are valid
        with open(grpc_json_file, "r") as f:
            grpc_data = json.load(f)
            assert isinstance(grpc_data, dict)
        
        with open(rest_json_file, "r") as f:
            rest_data = json.load(f)
            assert isinstance(rest_data, dict)
