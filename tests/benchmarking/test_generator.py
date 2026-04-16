"""
Tests for benchmarking/generator.py
Testing edge cases, error handling, and integration scenarios with mocking.
"""

import sys
import os
from unittest.mock import patch, mock_open, MagicMock, call, ANY
import pytest
import json
import numpy as np

# Add project root to path to import benchmarking module
# Since benchmarking is not a proper package, we need to add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from benchmarking import generator
from mlserver import types


class TestGenerateTestRequestsEdgeCases:
    """Test edge cases and error conditions in generate_test_requests."""
    
    def test_request_data_statistical_properties(self):
        """Verify generated data has expected statistical distribution."""
        np.random.seed(42)  # Set seed for reproducibility
        requests = generator.generate_test_requests()
        
        for req in requests:
            data_values = req.inputs[0].data.__root__
            data_array = np.array(data_values)
            
            # All values should be in range [0, 9999]
            assert np.all(data_array >= 0)
            assert np.all(data_array <= 9999)
            
            # Mean should be roughly around 4999.5 for uniform distribution
            mean_value = np.mean(data_array)
            assert 4000 < mean_value < 6000, f"Mean {mean_value} outside expected range"
    
    def test_requests_have_increasing_sizes(self):
        """Verify that requests have exponentially increasing sizes."""
        requests = generator.generate_test_requests()
        
        sizes = [req.inputs[0].shape[0] for req in requests]
        
        # Each size should be double the previous
        for i in range(len(sizes) - 1):
            assert sizes[i+1] == sizes[i] * 2
    
    def test_all_requests_have_fp32_datatype(self):
        """Ensure all generated requests use FP32 datatype."""
        requests = generator.generate_test_requests()
        
        for req in requests:
            assert req.inputs[0].datatype == "FP32"
    
    def test_tensor_data_serialization(self):
        """Test that TensorData properly validates the generated data."""
        requests = generator.generate_test_requests()
        
        for req in requests:
            # TensorData should be properly constructed
            assert hasattr(req.inputs[0].data, '__root__')
            assert isinstance(req.inputs[0].data.__root__, list)
    
    def test_memory_efficiency_for_large_arrays(self):
        """Test that large array generation doesn't cause memory issues."""
        # This should not raise MemoryError
        try:
            requests = generator.generate_test_requests()
            largest_request = requests[-1]
            assert largest_request.inputs[0].shape[0] == 2**15
        except MemoryError:
            pytest.fail("Memory error when generating test requests")


class TestSaveGrpcRequestsWithMocking:
    """Test save_grpc_requests with mocked file operations and error handling."""
    
    @patch('benchmarking.generator.open', new_callable=mock_open)
    @patch('benchmarking.generator.converters.ModelInferRequestConverter.from_types')
    def test_writes_protobuf_binary_with_size_prefix(self, mock_converter, mock_file):
        """Test that protobuf messages are prefixed with size varint."""
        mock_proto = MagicMock()
        mock_proto.ByteSize.return_value = 100
        mock_proto.SerializeToString.return_value = b'test_data'
        mock_converter.return_value = mock_proto
        
        requests = [MagicMock(spec=types.InferenceRequest)]
        generator.save_grpc_requests(requests)
        
        # Should write size varint and serialized data
        file_handle = mock_file()
        assert file_handle.write.call_count >= 2  # At least varint + data
    
    @patch('benchmarking.generator.open', new_callable=mock_open)
    @patch('benchmarking.generator.converters.ModelInferRequestConverter.from_types')
    @patch('benchmarking.generator.json_format.MessageToDict')
    @patch('benchmarking.generator.json.dump')
    def test_saves_both_pb_and_json_files(self, mock_json_dump, mock_msg_to_dict, mock_converter, mock_file):
        """Verify both .pb and .json files are created."""
        mock_proto = MagicMock()
        mock_proto.ByteSize.return_value = 50
        mock_proto.SerializeToString.return_value = b'data'
        mock_converter.return_value = mock_proto
        mock_msg_to_dict.return_value = {"test": "data"}
        
        requests = [MagicMock(spec=types.InferenceRequest)]
        generator.save_grpc_requests(requests)
        
        # open should be called twice: once for .pb, once for .json
        assert mock_file.call_count == 2
        mock_json_dump.assert_called_once()
    
    @patch('benchmarking.generator.open', side_effect=PermissionError("Access denied"))
    def test_handles_file_permission_errors(self, mock_file):
        """Test behavior when file write permission is denied."""
        requests = [MagicMock(spec=types.InferenceRequest)]
        
        with pytest.raises(PermissionError):
            generator.save_grpc_requests(requests)
    
    @patch('benchmarking.generator.open', new_callable=mock_open)
    @patch('benchmarking.generator.converters.ModelInferRequestConverter.from_types')
    def test_only_processes_first_request(self, mock_converter, mock_file):
        """Ensure only the first request is converted for gRPC."""
        mock_proto = MagicMock()
        mock_proto.ByteSize.return_value = 10
        mock_proto.SerializeToString.return_value = b'x'
        mock_converter.return_value = mock_proto
        
        requests = [MagicMock(spec=types.InferenceRequest) for _ in range(5)]
        generator.save_grpc_requests(requests)
        
        # Should only convert the first request
        assert mock_converter.call_count == 1
    
    @patch('benchmarking.generator.open', new_callable=mock_open)
    @patch('benchmarking.generator.converters.ModelInferRequestConverter.from_types')
    def test_uses_correct_model_name_and_version(self, mock_converter, mock_file):
        """Verify correct model name and version are passed to converter."""
        mock_proto = MagicMock()
        mock_proto.ByteSize.return_value = 10
        mock_proto.SerializeToString.return_value = b'x'
        mock_converter.return_value = mock_proto
        
        requests = [MagicMock(spec=types.InferenceRequest)]
        generator.save_grpc_requests(requests)
        
        mock_converter.assert_called_with(
            requests[0],
            model_name=generator.MODEL_NAME,
            model_version=generator.MODEL_VERSION
        )


class TestSaveRestRequestsErrorHandling:
    """Test save_rest_requests with focus on error scenarios and edge cases."""
    
    @patch('benchmarking.generator.open', new_callable=mock_open)
    def test_serializes_first_request_to_json(self, mock_file):
        """Test that first request is properly serialized to JSON."""
        mock_request = MagicMock(spec=types.InferenceRequest)
        mock_request.model_dump.return_value = {"inputs": [{"name": "test"}]}
        
        generator.save_rest_requests([mock_request])
        
        mock_request.model_dump.assert_called_once()
        file_handle = mock_file()
        assert file_handle.write.called
    
    @patch('benchmarking.generator.open', new_callable=mock_open)
    @patch('benchmarking.generator.json.dump')
    def test_json_dump_called_with_correct_params(self, mock_json_dump, mock_file):
        """Verify json.dump is called with request dict and file handle."""
        mock_request = MagicMock(spec=types.InferenceRequest)
        expected_dict = {"inputs": [{"name": "input-0"}]}
        mock_request.model_dump.return_value = expected_dict
        
        generator.save_rest_requests([mock_request])
        
        mock_json_dump.assert_called_once_with(expected_dict, mock_file())
    
    @patch('benchmarking.generator.open', side_effect=IOError("Disk full"))
    def test_handles_io_errors(self, mock_file):
        """Test behavior when disk is full or IO error occurs."""
        requests = [MagicMock(spec=types.InferenceRequest)]
        
        with pytest.raises(IOError):
            generator.save_rest_requests(requests)
    
    @patch('benchmarking.generator.open', new_callable=mock_open)
    def test_writes_to_correct_file_path(self, mock_file):
        """Verify the correct file path is used for REST requests."""
        mock_request = MagicMock(spec=types.InferenceRequest)
        mock_request.model_dump.return_value = {}
        
        generator.save_rest_requests([mock_request])
        
        expected_path = os.path.join(generator.DATA_PATH, "rest-requests.json")
        mock_file.assert_called_with(expected_path, "w")
    
    def test_handles_empty_request_list_gracefully(self):
        """Test behavior with empty request list."""
        # This should raise IndexError since we access requests[0]
        with pytest.raises(IndexError):
            generator.save_rest_requests([])


class TestMainFunctionIntegration:
    """Integration tests for main() with mocked dependencies."""
    
    @patch('benchmarking.generator.save_rest_requests')
    @patch('benchmarking.generator.save_grpc_requests')
    @patch('benchmarking.generator.generate_test_requests')
    def test_main_calls_all_functions_in_order(self, mock_gen, mock_grpc, mock_rest):
        """Test that main calls functions in the correct sequence."""
        mock_requests = [MagicMock()]
        mock_gen.return_value = mock_requests
        
        generator.main()
        
        mock_gen.assert_called_once()
        mock_grpc.assert_called_once_with(mock_requests)
        mock_rest.assert_called_once_with(mock_requests)
    
    @patch('benchmarking.generator.save_rest_requests')
    @patch('benchmarking.generator.save_grpc_requests')
    @patch('benchmarking.generator.generate_test_requests', side_effect=Exception("Generation failed"))
    def test_main_propagates_generation_errors(self, mock_gen, mock_grpc, mock_rest):
        """Test that errors in request generation are propagated."""
        with pytest.raises(Exception, match="Generation failed"):
            generator.main()
        
        # Neither save function should be called if generation fails
        mock_grpc.assert_not_called()
        mock_rest.assert_not_called()
    
    @patch('benchmarking.generator.save_rest_requests', side_effect=IOError("Write failed"))
    @patch('benchmarking.generator.save_grpc_requests')
    @patch('benchmarking.generator.generate_test_requests')
    def test_main_propagates_save_errors(self, mock_gen, mock_grpc, mock_rest):
        """Test that errors in save operations are propagated."""
        mock_gen.return_value = [MagicMock()]
        
        with pytest.raises(IOError, match="Write failed"):
            generator.main()
    
    @patch('benchmarking.generator.save_rest_requests')
    @patch('benchmarking.generator.save_grpc_requests')
    @patch('benchmarking.generator.generate_test_requests')
    def test_main_passes_same_requests_to_both_savers(self, mock_gen, mock_grpc, mock_rest):
        """Verify the same request list is passed to both save functions."""
        mock_requests = [MagicMock(), MagicMock()]
        mock_gen.return_value = mock_requests
        
        generator.main()
        
        # Both save functions should receive the same request list
        assert mock_grpc.call_args[0][0] is mock_requests
        assert mock_rest.call_args[0][0] is mock_requests


class TestParametrizedScenarios:
    """Parametrized tests for various input scenarios."""
    
    @pytest.mark.parametrize("seed", [0, 42, 123, 999, 12345])
    def test_generate_requests_with_different_seeds(self, seed):
        """Test request generation with different random seeds."""
        np.random.seed(seed)
        requests = generator.generate_test_requests()
        
        assert len(requests) == 6
        for req in requests:
            assert req.inputs[0].datatype == "FP32"
            data = req.inputs[0].data.__root__
            assert all(0 <= val <= 9999 for val in data)
    
    @pytest.mark.parametrize("index,expected_size", [
        (0, 2**10),
        (1, 2**11),
        (2, 2**12),
        (3, 2**13),
        (4, 2**14),
        (5, 2**15),
    ])
    def test_request_sizes_at_each_index(self, index, expected_size):
        """Test that each request has the expected size."""
        requests = generator.generate_test_requests()
        assert requests[index].inputs[0].shape[0] == expected_size
    
    @pytest.mark.parametrize("file_mode,file_extension", [
        ("wb", ".pb"),
        ("w", ".json"),
    ])
    @patch('benchmarking.generator.open', new_callable=mock_open)
    @patch('benchmarking.generator.converters.ModelInferRequestConverter.from_types')
    @patch('benchmarking.generator.json_format.MessageToDict', return_value={})
    @patch('benchmarking.generator.json.dump')
    def test_grpc_file_modes(self, mock_dump, mock_dict, mock_conv, mock_file, file_mode, file_extension):
        """Test that files are opened with correct modes."""
        mock_proto = MagicMock()
        mock_proto.ByteSize.return_value = 10
        mock_proto.SerializeToString.return_value = b'x'
        mock_conv.return_value = mock_proto
        
        requests = [MagicMock(spec=types.InferenceRequest)]
        generator.save_grpc_requests(requests)
        
        # Check that appropriate mode was used for file operations
        calls = mock_file.call_args_list
        modes_used = [call[0][1] if len(call[0]) > 1 else call[1].get('mode', 'r') for call in calls]
        assert file_mode in modes_used


class TestDataPathConfiguration:
    """Test DATA_PATH configuration and usage."""
    
    def test_data_path_exists_in_module(self):
        """Verify DATA_PATH is properly configured."""
        assert hasattr(generator, 'DATA_PATH')
        assert isinstance(generator.DATA_PATH, str)
    
    def test_data_path_contains_sum_model(self):
        """Verify DATA_PATH points to sum-model directory."""
        assert "sum-model" in generator.DATA_PATH
    
    def test_model_name_and_version_constants(self):
        """Verify model constants are defined."""
        assert generator.MODEL_NAME == "sum-model"
        assert generator.MODEL_VERSION == "v1.2.3"
    
    @patch('benchmarking.generator.open', new_callable=mock_open)
    @patch('benchmarking.generator.os.path.join')
    def test_save_functions_use_data_path(self, mock_join, mock_file):
        """Test that save functions properly use DATA_PATH."""
        mock_join.return_value = "/fake/path/file.json"
        mock_request = MagicMock(spec=types.InferenceRequest)
        mock_request.model_dump.return_value = {}
        
        generator.save_rest_requests([mock_request])
        
        # os.path.join should be called with DATA_PATH
        mock_join.assert_called()
        call_args = mock_join.call_args[0]
        assert generator.DATA_PATH in call_args


class TestNumpyArrayHandling:
    """Test numpy array generation and manipulation."""
    
    def test_numpy_random_usage(self):
        """Test that numpy random is used correctly."""
        np.random.seed(42)
        requests = generator.generate_test_requests()
        
        # Verify we can extract numpy-compatible data
        for req in requests:
            data = req.inputs[0].data.__root__
            arr = np.array(data)
            assert arr.dtype in [np.float64, np.float32, float]
    
    def test_contents_lens_calculation(self):
        """Verify the calculation of content lengths."""
        # contents_lens = np.power(2, np.arange(10, 16)).astype(int)
        expected = [2**i for i in range(10, 16)]
        
        requests = generator.generate_test_requests()
        actual = [req.inputs[0].shape[0] for req in requests]
        
        assert actual == expected
    
    def test_random_data_not_all_same(self):
        """Ensure generated random data has variance."""
        np.random.seed(42)
        requests = generator.generate_test_requests()
        
        for req in requests:
            data = req.inputs[0].data.__root__
            # Check that not all values are identical
            assert len(set(data)) > 1, "Random data should have variance"
    
    def test_array_conversion_to_list(self):
        """Test that numpy arrays are properly converted to lists."""
        requests = generator.generate_test_requests()
        
        for req in requests:
            data = req.inputs[0].data.__root__
            assert isinstance(data, list), "Data should be converted to list"
            assert all(isinstance(x, (int, float)) for x in data)
