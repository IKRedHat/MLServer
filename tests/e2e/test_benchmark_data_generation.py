import os
import json

DATA_PATH = "benchmarking/data/sum-model"

def test_benchmark_data_generation():
    # Check if the directory exists
    assert os.path.exists(DATA_PATH), f"Directory {DATA_PATH} does not exist"

    # Check if the files exist
    grpc_pb_file = os.path.join(DATA_PATH, "grpc-requests.pb")
    grpc_json_file = os.path.join(DATA_PATH, "grpc-requests.json")
    rest_json_file = os.path.join(DATA_PATH, "rest-requests.json")

    assert os.path.exists(grpc_pb_file), f"File {grpc_pb_file} does not exist"
    assert os.path.exists(grpc_json_file), f"File {grpc_json_file} does not exist"
    assert os.path.exists(rest_json_file), f"File {rest_json_file} does not exist"

    # Check if the files are not empty
    assert os.path.getsize(grpc_pb_file) > 0, f"File {grpc_pb_file} is empty"
    assert os.path.getsize(grpc_json_file) > 0, f"File {grpc_json_file} is empty"
    assert os.path.getsize(rest_json_file) > 0, f"File {rest_json_file} is empty"