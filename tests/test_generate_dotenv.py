import os
import json
import sys
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import pytest
from click.testing import CliRunner

# Ensure the root directory is in sys.path so we can import hack and mlserver
sys.path.insert(0, str(Path(__file__).parent.parent))

from hack.generate_dotenv import (
    load_default_settings,
    _read_json_file,
    get_default_env,
    _convert_to_env,
    _get_env_prefix,
    save_default_env,
    _parse_dict_values,
    main,
)
from mlserver.settings import Settings, ModelSettings, ModelParameters


def test_read_json_file():
    mock_data = '{"key": "value"}'
    with patch("builtins.open", mock_open(read_data=mock_data)):
        result = _read_json_file("dummy.json")
        assert result == {"key": "value"}


def test_read_json_file_invalid_json():
    mock_data = '{"key": "value"'
    with patch("builtins.open", mock_open(read_data=mock_data)):
        with pytest.raises(json.JSONDecodeError):
            _read_json_file("dummy.json")


@patch("hack.generate_dotenv.os.path.isfile")
@patch("hack.generate_dotenv._read_json_file")
def test_load_default_settings_both_files(mock_read_json, mock_isfile):
    mock_isfile.side_effect = lambda x: True
    
    def read_json_side_effect(path):
        if "model-settings.json" in path:
            return {"name": "my-model", "parameters": {"version": "v1"}}
        return {"http_port": 8080}
        
    mock_read_json.side_effect = read_json_side_effect
    
    settings = load_default_settings("/dummy/folder")
    
    assert len(settings) == 3
    assert settings[0] == (Settings, {"http_port": 8080})
    assert settings[1] == (ModelSettings, {"name": "my-model"})
    assert settings[2] == (ModelParameters, {"version": "v1"})


@patch("hack.generate_dotenv.os.path.isfile")
@patch("hack.generate_dotenv._read_json_file")
def test_load_default_settings_no_files(mock_read_json, mock_isfile):
    mock_isfile.return_value = False
    settings = load_default_settings("/dummy/folder")
    assert len(settings) == 0


@patch("hack.generate_dotenv.os.path.isfile")
@patch("hack.generate_dotenv._read_json_file")
def test_load_default_settings_only_settings(mock_read_json, mock_isfile):
    mock_isfile.side_effect = lambda x: "model-settings.json" not in x
    mock_read_json.return_value = {"http_port": 8080}
    
    settings = load_default_settings("/dummy/folder")
    assert len(settings) == 1
    assert settings[0] == (Settings, {"http_port": 8080})


@patch("hack.generate_dotenv.os.path.isfile")
@patch("hack.generate_dotenv._read_json_file")
def test_load_default_settings_only_model_settings_no_params(mock_read_json, mock_isfile):
    mock_isfile.side_effect = lambda x: "model-settings.json" in x
    mock_read_json.return_value = {"name": "my-model"}
    
    settings = load_default_settings("/dummy/folder")
    assert len(settings) == 1
    assert settings[0] == (ModelSettings, {"name": "my-model"})


@patch("hack.generate_dotenv.os.path.isfile")
@patch("hack.generate_dotenv._read_json_file")
def test_load_default_settings_null_parameters(mock_read_json, mock_isfile):
    mock_isfile.side_effect = lambda x: "model-settings.json" in x
    mock_read_json.return_value = {"name": "my-model", "parameters": None}
    
    settings = load_default_settings("/dummy/folder")
    assert len(settings) == 1
    assert settings[0] == (ModelSettings, {"name": "my-model"})


@patch("hack.generate_dotenv.os.path.isfile")
def test_load_default_settings_file_deleted(mock_isfile):
    mock_isfile.return_value = True
    with patch("builtins.open", side_effect=FileNotFoundError("File deleted")):
        with pytest.raises(FileNotFoundError):
            load_default_settings("/dummy/folder")


def test_get_env_prefix():
    class MockConfig:
        env_prefix = "MOCK_"
        
    class MockSettings:
        Config = MockConfig
        
    assert _get_env_prefix(MockSettings) == "MOCK_"
    
    class MockSettingsNoPrefix:
        Config = object()
        
    assert _get_env_prefix(MockSettingsNoPrefix) == ""
    
    class MockSettingsNoConfig:
        pass
        
    assert _get_env_prefix(MockSettingsNoConfig) == ""


def test_convert_to_env():
    class MockConfig:
        env_prefix = "TEST_"
        
    class MockSettings:
        Config = MockConfig
        
    raw_defaults = {"port": 8080, "host": "localhost"}
    env = _convert_to_env(MockSettings, raw_defaults)
    
    assert env == {
        "TEST_PORT": "8080",
        "TEST_HOST": "localhost"
    }


def test_get_default_env():
    class MockConfig1:
        env_prefix = "S1_"
    class MockSettings1:
        Config = MockConfig1
        
    class MockConfig2:
        env_prefix = "S2_"
    class MockSettings2:
        Config = MockConfig2
        
    default_settings = [
        (MockSettings1, {"a": 1}),
        (MockSettings2, {"b": 2})
    ]
    
    env = get_default_env(default_settings)
    assert env == {
        "S1_A": "1",
        "S2_B": "2"
    }


def test_parse_dict_values():
    # Valid JSON string (dict)
    assert _parse_dict_values("VAR1", "{'key': 'value'}") == "VAR1='{\"key\": \"value\"}'\n"
    
    # Valid JSON string (list)
    assert _parse_dict_values("VAR2", "['a', 'b']") == "VAR2='[\"a\", \"b\"]'\n"
    
    # Valid JSON primitive
    assert _parse_dict_values("VAR3", "123") == "VAR3='123'\n"
    
    # Plain string (not valid JSON)
    assert _parse_dict_values("VAR4", "plain_string") == 'VAR4="plain_string"\n'


def test_save_default_env():
    env = {
        "VAR1": "{'key': 'value'}",
        "VAR2": "plain_string"
    }
    
    m_open = mock_open()
    with patch("builtins.open", m_open):
        save_default_env(env, "output.env")
        
    m_open.assert_called_once_with("output.env", "w")
    handle = m_open()
    handle.write.assert_any_call("VAR1='{\"key\": \"value\"}'\n")
    handle.write.assert_any_call('VAR2="plain_string"\n')


def test_save_default_env_io_error():
    env = {"VAR1": "value"}
    with patch("builtins.open", side_effect=IOError("Permission denied")):
        with pytest.raises(IOError):
            save_default_env(env, "/root/output.env")


@patch("hack.generate_dotenv.save_default_env")
@patch("hack.generate_dotenv.get_default_env")
@patch("hack.generate_dotenv.load_default_settings")
def test_main_cli(mock_load, mock_get, mock_save):
    mock_load.return_value = []
    mock_get.return_value = {"VAR": "VAL"}
    
    runner = CliRunner()
    result = runner.invoke(main, ["/dummy/folder", "output.env"])
    
    assert result.exit_code == 0
    mock_load.assert_called_once_with("/dummy/folder")
    mock_get.assert_called_once_with([])
    mock_save.assert_called_once_with({"VAR": "VAL"}, "output.env")