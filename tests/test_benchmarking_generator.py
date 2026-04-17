"""
Comprehensive unit tests for benchmarking/generator.py module.
Tests focus on mathematical properties, boundary conditions, and error handling.
"""

import sys
import os
import pytest
import numpy as np
import json
from unittest.mock import patch, mock_open, MagicMock, call
from pathlib import Path

# Import the generator module by adding its parent directory to sys.path
generator_path = Path(__file__).parent.parent / "benchmarking"
sys.path.insert(0, str(generator_path.parent))

try:
    import benchmarking.generator as generator
except ImportError:
    # Fallback: try importing from current directory structure
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generator", 
        Path(__file__).parent.parent / "benchmarking" / "generator.py"
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)


class TestDataGenerationMathematics:
    """Test mathematical properties and numerical correctness of data generation."""
    
    def test_exponential_size_progression(self):
        """Verify array sizes follow 2^n progression from 10 to 15."""
        requests = generator.generate_test_requests()
        
        expected_powers = list(range(10, 16))
        expected_sizes = [2**power for power in expected_powers]
        actual_sizes = [req.inputs[0].shape[0] for req in requests]
        
        assert actual_sizes == expected_sizes
        assert len(requests) == 6
    
    def test_random_data_distribution_properties(self):
        """Test statistical properties of generated random data."""
        np.random.seed(12345)  # Fixed seed for reproducible tests
        requests = generator.generate_test_requests()
        
        for i, req in enumerate(requests):
            data_values = req.inputs[0].data.__root__
            data_array = np.array(data_values)
            
            # Test range constraints
            assert np.all(data_array >= 0), f"Request {i}: negative values found"
            assert np.all(data_array < 9999), f"Request {i}: values exceed max_value"
            
            # Test distribution properties for larger arrays
            if len(data_array) >= 1024:
                mean_val = np.mean(data_array)
                std_val = np.std(data_array)
                
                # For uniform distribution [0, 9999), mean ≈ 4999.5, std ≈ 2886
                assert 4000 < mean_val < 6000, f"Request {i}: mean {mean_val} outside expected range"
                assert 2000 < std_val < 4000, f"Request {i}: std {std_val} outside expected range"
    
    def test_data_type_consistency(self):
        """Verify all generated data maintains FP32 type consistency."""
        requests = generator.generate_test_requests()
        
        for req in requests:
            assert req.inputs[0].datatype == "FP32"
            assert req.inputs[0].name == "input-0"
            assert len(req.inputs) == 1
            assert len(req.inputs[0].shape) == 1
    
    def test_tensor_data_validation(self):
        """Test TensorData model validation with edge cases."""
        requests = generator.generate_test_requests()
        
        for req in requests:
            tensor_data = req.inputs[0].data
            assert hasattr(tensor_data, '__root__')
            assert isinstance(tensor_data.__root__, list)
            assert len(tensor_data.__root__) == req.inputs[0].shape[0]
    
    def test_memory_bounds_largest_array(self):
        """Test memory efficiency for largest generated array (2^15 elements)."""
        requests = generator.generate_test_requests()
        largest_request = requests[-1]
        
        assert largest_request.inputs[0].shape[0] == 32768
        data_size = len(largest_request.inputs[0].data.__root__)
        assert data_size == 32768
        
        # Verify memory usage is reasonable (each float ~8 bytes in Python)
        import sys
        data_memory = sys.getsizeof(largest_request.inputs[0].data.__root__)
        assert data_memory < 1024 * 1024  # Less than 1MB


class TestProtobufSerializationLogic:
    """Test gRPC protobuf serialization with focus on binary format correctness."""
    
    @patch('benchmarking.generator.open', new_callable=mock_open)
    @patch('benchmarking.generator.converters.ModelInferRequestConverter.from_types')
    def test_varint_size_encoding_accuracy(self, mock_converter, mock_file):
        """Test that varint size prefixes are correctly calculated and written."""
        mock_proto = MagicMock()
        mock_proto.ByteSize.return_value = 127  # Single-byte varint
        mock_proto.SerializeToString.return_value = b'x' * 127
        mock_converter.return_value = mock_proto
        
        requests = [MagicMock()]
        generator.save_grpc_requests(requests)
        
        # Verify varint encoding for size 127 (single byte: 0x7F)
        file_handle = mock_file()
        write_calls = file_handle.write.call_args_list
        
        assert len(write_calls) >= 2  # Size + data
        size_bytes = write_calls[0][0][0]
        assert size_bytes == b'\x7f'  # 127 as single-byte varint
    
    @patch('benchmarking.generator.open', new_callable=mock_open)
    @patch('benchmarking.generator.converters.ModelInferRequestConverter.from_types')
    def test_multi_byte_varint_encoding(self, mock_converter, mock_file):
        """Test varint encoding for sizes requiring multiple bytes."""
        mock_proto = MagicMock()
        mock_proto.ByteSize.return_value = 300  # Multi-byte varint
        mock_proto.SerializeToString.return_value = b'x' * 300
        mock_converter.return_value = mock_proto
        
        requests = [MagicMock()]
        generator.save_grpc_requests(requests)
        
        file_handle = mock_file()
        write_calls = file_handle.write.call_args_list
        
        # 300 = 0xAC 0x02 in varint encoding
        size_bytes = write_calls[0][0][0]
        assert len(size_bytes) == 2  # Multi-byte varint
        assert size_bytes[0] & 0x80 != 0  # Continuation bit set
    
    @patch('benchmarking.generator.open', new_callable=mock_open)
    @patch('benchmarking.generator.converters.ModelInferRequestConverter.from_types')
    @patch('benchmarking.generator.json_format.MessageToDict')
    def test_protobuf_to_json_conversion_integrity(self, mock_msg_to_dict, mock_converter, mock_file):
        """Test protobuf to JSON conversion maintains data integrity."""
        mock_proto = MagicMock()
        mock_proto.ByteSize.return_value = 50
        mock_proto.SerializeToString.return_value = b'data'
        mock_converter.return_value = mock_proto
        
        expected_dict = {
            "modelName": "sum-model",
            "modelVersion": "v1.2.3",
            "inputs": [{"name": "input-0", "datatype": "FP32"}]
        }
        mock_msg_to_dict.return_value = expected_dict
        
        requests = [MagicMock()]
        generator.save_grpc_requests(requests)
        
        mock_msg_to_dict.assert_called_once_with(mock_proto)
    
    @patch('benchmarking.generator.converters.ModelInferRequestConverter.from_types')
    def test_converter_called_with_correct_parameters(self, mock_converter):
        """Verify converter receives correct model name and version."""
        mock_proto = MagicMock()
        mock_proto.ByteSize.return_value = 10
        mock_proto.SerializeToString.return_value = b'x'
        mock_converter.return_value = mock_proto
        
        with patch('benchmarking.generator.open', mock_open()):
            requests = [MagicMock()]
            generator.save_grpc_requests(requests)
        
        mock_converter.assert_called_once_with(
            requests[0],
            model_name="sum-model",
            model_version="v1.2.3"
        )


class TestJSONSerializationValidation:
    """Test REST JSON serialization with schema validation and type preservation."""
    
    @patch('benchmarking.generator.open', new_callable=mock_open)
    @patch('benchmarking.generator.json.dump')
    def test_json_schema_structure_validation(self, mock_json_dump, mock_file):
        """Test that JSON output conforms to expected schema structure."""
        mock_request = MagicMock()
        expected_dict = {
            "inputs": [{
                "name": "input-0",
                "shape": [1024],
                "datatype": "FP32",
                "data": list(range(1024))
            }]
        }
        mock_request.model_dump.return_value = expected_dict
        
        generator.save_rest_requests([mock_request])
        
        mock_json_dump.assert_called_once_with(expected_dict, mock_file())
        mock_request.model_dump.assert_called_once()
    
    @patch('benchmarking.generator.open', new_callable=mock_open)
    def test_first_request_selection_logic(self, mock_file):
        """Verify only the first request is processed for REST output."""
        requests = [MagicMock() for _ in range(5)]
        requests[0].model_dump.return_value = {"test": "first"}
        
        generator.save_rest_requests(requests)
        
        # Only first request should be called
        requests[0].model_dump.assert_called_once()
        for req in requests[1:]:
            req.model_dump.assert_not_called()
    
    @patch('benchmarking.generator.open', side_effect=OSError("Disk full"))
    def test_json_file_write_error_handling(self, mock_file):
        """Test behavior when JSON file write fails due to disk space."""
        requests = [MagicMock()]
        requests[0].model_dump.return_value = {"data": "test"}
        
        with pytest.raises(OSError, match="Disk full"):
            generator.save_rest_requests(requests)
    
    @patch('benchmarking.generator.open', new_callable=mock_open)
    @patch('benchmarking.generator.json.dump', side_effect=ValueError("Invalid JSON"))
    def test_json_serialization_error_handling(self, mock_json_dump, mock_file):
        """Test handling of JSON serialization errors."""
        requests = [MagicMock()]
        requests[0].model_dump.return_value = {"invalid": float('inf')}
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            generator.save_rest_requests(requests)


class TestFileSystemOperations:
    """Test file system operations, path handling, and atomic operations."""
    
    @patch('benchmarking.generator.os.path.join')
    def test_data_path_construction(self, mock_join):
        """Test that file paths are constructed correctly using DATA_PATH."""
        mock_join.side_effect = lambda *args: "/".join(args)
        
        with patch('benchmarking.generator.open', mock_open()):
            with patch('benchmarking.generator.converters.ModelInferRequestConverter.from_types') as mock_conv:
                mock_proto = MagicMock()
                mock_proto.ByteSize.return_value = 10
                mock_proto.SerializeToString.return_value = b'x'
                mock_conv.return_value = mock_proto
                
                generator.save_grpc_requests([MagicMock()])
        
        # Verify path construction calls
        expected_calls = [
            call(generator.DATA_PATH, "grpc-requests.pb"),
            call(generator.DATA_PATH, "grpc-requests.json")
        ]
        mock_join.assert_has_calls(expected_calls, any_order=True)
    
    @patch('benchmarking.generator.open', side_effect=[
        mock_open().return_value,  # First file succeeds
        PermissionError("Access denied")  # Second file fails
    ])
    @patch('benchmarking.generator.converters.ModelInferRequestConverter.from_types')
    def test_partial_file_write_failure(self, mock_converter, mock_file):
        """Test behavior when one file writes successfully but another fails."""
        mock_proto = MagicMock()
        mock_proto.ByteSize.return_value = 10
        mock_proto.SerializeToString.return_value = b'x'
        mock_converter.return_value = mock_proto
        
        with pytest.raises(PermissionError):
            generator.save_grpc_requests([MagicMock()])
    
    @patch('benchmarking.generator.generate_test_requests')
    @patch('benchmarking.generator.save_grpc_requests')
    @patch('benchmarking.generator.save_rest_requests')
    def test_main_function_orchestration(self, mock_save_rest, mock_save_grpc, mock_generate):
        """Test main function calls all components in correct order."""
        mock_requests = [MagicMock(), MagicMock()]
        mock_generate.return_value = mock_requests
        
        generator.main()
        
        mock_generate.assert_called_once()
        mock_save_grpc.assert_called_once_with(mock_requests)
        mock_save_rest.assert_called_once_with(mock_requests)
    
    @patch('benchmarking.generator.generate_test_requests', side_effect=MemoryError("Out of memory"))
    def test_main_function_memory_error_propagation(self, mock_generate):
        """Test that memory errors in generation are properly propagated."""
        with pytest.raises(MemoryError, match="Out of memory"):
            generator.main()


class TestModuleConstants:
    """Test module-level constants and configuration validation."""
    
    def test_model_constants_immutability(self):
        """Verify model name and version constants have expected values."""
        assert generator.MODEL_NAME == "sum-model"
        assert generator.MODEL_VERSION == "v1.2.3"
        assert isinstance(generator.MODEL_NAME, str)
        assert isinstance(generator.MODEL_VERSION, str)
    
    def test_data_path_structure(self):
        """Test DATA_PATH construction and directory structure."""
        expected_components = ["data", "sum-model"]
        
        # DATA_PATH should end with the expected components
        path_parts = generator.DATA_PATH.split(os.sep)
        assert path_parts[-2:] == expected_components
    
    @patch('benchmarking.generator.os.path.dirname')
    @patch('benchmarking.generator.os.path.join')
    def test_data_path_construction_logic(self, mock_join, mock_dirname):
        """Test the logic used to construct DATA_PATH."""
        mock_dirname.return_value = "/path/to/benchmarking"
        mock_join.return_value = "/path/to/benchmarking/data/sum-model"
        
        # Reload the module to trigger path construction
        import importlib
        importlib.reload(generator)
        
        mock_dirname.assert_called()
        mock_join.assert_called_with(mock_dirname.return_value, "data", "sum-model")


class TestNumpyIntegration:
    """Test numpy operations and numerical edge cases."""
    
    def test_power_array_generation(self):
        """Test np.power and np.arange operations for size calculation."""
        # Replicate the logic from generate_test_requests
        contents_lens = np.power(2, np.arange(10, 16)).astype(int)
        
        expected = [1024, 2048, 4096, 8192, 16384, 32768]
        assert contents_lens.tolist() == expected
        assert contents_lens.dtype == int
    
    def test_random_array_scaling(self):
        """Test random array generation and scaling operations."""
        np.random.seed(42)
        max_value = 9999
        contents_len = 1000
        
        inputs = max_value * np.random.rand(contents_len)
        
        assert len(inputs) == contents_len
        assert np.all(inputs >= 0)
        assert np.all(inputs < max_value)
        assert inputs.dtype == np.float64  # Default numpy float type
    
    @patch('benchmarking.generator.np.random.rand', side_effect=MemoryError("Array too large"))
    def test_numpy_memory_error_handling(self, mock_rand):
        """Test handling of numpy memory errors during array generation."""
        with pytest.raises(MemoryError):
            generator.generate_test_requests()
    
    def test_array_to_list_conversion(self):
        """Test numpy array to Python list conversion for TensorData."""
        test_array = np.array([1.0, 2.0, 3.0])
        list_data = test_array.tolist()
        
        assert isinstance(list_data, list)
        assert all(isinstance(x, float) for x in list_data)
        assert list_data == [1.0, 2.0, 3.0]