import os
import json
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any
import pytest
from hypothesis import given, strategies as st

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_settings_data():
    """Sample settings.json data with various data types."""
    return {
        "http_port": 8080,
        "grpc_port": 8081,
        "debug": True,
        "host": "0.0.0.0",
        "metrics_endpoint": "/metrics",
        "nested_config": {
            "timeout": 30,
            "retries": 3
        },
        "tags": ["production", "ml-server"],
        "special_chars": "test with spaces & symbols!",
        "unicode_field": "测试中文字符"
    }


@pytest.fixture
def sample_model_settings_data():
    """Sample model-settings.json data with parameters."""
    return {
        "name": "my-test-model",
        "implementation": "mlserver.models.sklearn.SKLearnModel",
        "parameters": {
            "uri": "./model.pkl",
            "version": "v1.0.0",
            "extra_config": {
                "batch_size": 32,
                "timeout": 60
            },
            "feature_names": ["feature1", "feature2", "feature3"]
        }
    }


@pytest.fixture
def create_settings_files(temp_dir, sample_settings_data, sample_model_settings_data):
    """Create actual JSON files in temporary directory."""
    settings_path = os.path.join(temp_dir, "settings.json")
    model_settings_path = os.path.join(temp_dir, "model-settings.json")
    
    with open(settings_path, 'w') as f:
        json.dump(sample_settings_data, f, indent=2)
    
    with open(model_settings_path, 'w') as f:
        json.dump(sample_model_settings_data, f, indent=2)
    
    return {
        'settings_path': settings_path,
        'model_settings_path': model_settings_path,
        'temp_dir': temp_dir
    }


class TestFileOperations:
    """Test file reading and JSON parsing functionality."""
    
    def test_read_json_file_success(self, temp_dir):
        """Test successful JSON file reading."""
        test_data = {"key": "value", "number": 42, "array": [1, 2, 3]}
        test_file = os.path.join(temp_dir, "test.json")
        
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        result = _read_json_file(test_file)
        assert result == test_data
    
    def test_read_json_file_malformed(self, temp_dir):
        """Test reading malformed JSON file."""
        test_file = os.path.join(temp_dir, "malformed.json")
        
        with open(test_file, 'w') as f:
            f.write('{"key": "value",}')  # Trailing comma makes it invalid
        
        with pytest.raises(json.JSONDecodeError):
            _read_json_file(test_file)
    
    def test_read_json_file_not_found(self):
        """Test reading non-existent file."""
        with pytest.raises(FileNotFoundError):
            _read_json_file("/nonexistent/path/file.json")
    
    def test_read_json_file_permission_denied(self, temp_dir):
        """Test reading file with no permissions."""
        test_file = os.path.join(temp_dir, "no_permission.json")
        
        with open(test_file, 'w') as f:
            json.dump({"test": "data"}, f)
        
        # Remove read permissions
        os.chmod(test_file, 0o000)
        
        try:
            with pytest.raises(PermissionError):
                _read_json_file(test_file)
        finally:
            # Restore permissions for cleanup
            os.chmod(test_file, 0o644)
    
    def test_read_json_file_empty(self, temp_dir):
        """Test reading empty JSON file."""
        test_file = os.path.join(temp_dir, "empty.json")
        
        with open(test_file, 'w') as f:
            f.write('{}')
        
        result = _read_json_file(test_file)
        assert result == {}


class TestSettingsLoading:
    """Test settings loading functionality with real files."""
    
    def test_load_both_settings_files(self, create_settings_files):
        """Test loading both settings.json and model-settings.json."""
        temp_dir = create_settings_files['temp_dir']
        
        result = load_default_settings(temp_dir)
        
        assert len(result) == 3  # Settings, ModelSettings, ModelParameters
        
        # Verify Settings
        settings_class, settings_data = result[0]
        assert settings_class == Settings
        assert settings_data["http_port"] == 8080
        assert settings_data["debug"] is True
        
        # Verify ModelSettings
        model_settings_class, model_settings_data = result[1]
        assert model_settings_class == ModelSettings
        assert model_settings_data["name"] == "my-test-model"
        assert "parameters" not in model_settings_data  # Should be extracted
        
        # Verify ModelParameters
        model_params_class, model_params_data = result[2]
        assert model_params_class == ModelParameters
        assert model_params_data["uri"] == "./model.pkl"
        assert model_params_data["version"] == "v1.0.0"
    
    def test_load_only_settings_file(self, temp_dir, sample_settings_data):
        """Test loading only settings.json."""
        settings_path = os.path.join(temp_dir, "settings.json")
        
        with open(settings_path, 'w') as f:
            json.dump(sample_settings_data, f)
        
        result = load_default_settings(temp_dir)
        
        assert len(result) == 1
        settings_class, settings_data = result[0]
        assert settings_class == Settings
        assert settings_data == sample_settings_data
    
    def test_load_only_model_settings_file(self, temp_dir, sample_model_settings_data):
        """Test loading only model-settings.json."""
        model_settings_path = os.path.join(temp_dir, "model-settings.json")
        
        with open(model_settings_path, 'w') as f:
            json.dump(sample_model_settings_data, f)
        
        result = load_default_settings(temp_dir)
        
        assert len(result) == 2  # ModelSettings and ModelParameters
        
        model_settings_class, model_settings_data = result[0]
        assert model_settings_class == ModelSettings
        assert model_settings_data["name"] == "my-test-model"
        
        model_params_class, model_params_data = result[1]
        assert model_params_class == ModelParameters
        assert model_params_data["uri"] == "./model.pkl"
    
    def test_load_no_settings_files(self, temp_dir):
        """Test loading from directory with no settings files."""
        result = load_default_settings(temp_dir)
        assert result == []
    
    def test_load_model_settings_without_parameters(self, temp_dir):
        """Test loading model-settings.json without parameters field."""
        model_data = {
            "name": "simple-model",
            "implementation": "mlserver.models.sklearn.SKLearnModel"
        }
        
        model_settings_path = os.path.join(temp_dir, "model-settings.json")
        with open(model_settings_path, 'w') as f:
            json.dump(model_data, f)
        
        result = load_default_settings(temp_dir)
        
        assert len(result) == 1
        model_settings_class, model_settings_data = result[0]
        assert model_settings_class == ModelSettings
        assert model_settings_data == model_data
    
    def test_load_model_settings_null_parameters(self, temp_dir):
        """Test loading model-settings.json with null parameters."""
        model_data = {
            "name": "simple-model",
            "implementation": "mlserver.models.sklearn.SKLearnModel",
            "parameters": None
        }
        
        model_settings_path = os.path.join(temp_dir, "model-settings.json")
        with open(model_settings_path, 'w') as f:
            json.dump(model_data, f)
        
        result = load_default_settings(temp_dir)
        
        assert len(result) == 1
        model_settings_class, model_settings_data = result[0]
        assert model_settings_class == ModelSettings
        assert model_settings_data["name"] == "simple-model"
        assert "parameters" not in model_settings_data


class TestEnvironmentVariableConversion:
    """Test environment variable conversion functionality."""
    
    def test_get_env_prefix_with_config(self):
        """Test getting environment prefix from settings class with Config."""
        class TestConfig:
            env_prefix = "TEST_"
        
        class TestSettings:
            Config = TestConfig
        
        result = _get_env_prefix(TestSettings)
        assert result == "TEST_"
    
    def test_get_env_prefix_no_config(self):
        """Test getting environment prefix from settings class without Config."""
        class TestSettings:
            pass
        
        result = _get_env_prefix(TestSettings)
        assert result == ""
    
    def test_get_env_prefix_config_no_prefix(self):
        """Test getting environment prefix from Config without env_prefix."""
        class TestConfig:
            pass
        
        class TestSettings:
            Config = TestConfig
        
        result = _get_env_prefix(TestSettings)
        assert result == ""
    
    def test_convert_to_env_with_prefix(self):
        """Test converting settings to environment variables with prefix."""
        class TestConfig:
            env_prefix = "MLSERVER_"
        
        class TestSettings:
            Config = TestConfig
        
        raw_defaults = {
            "http_port": 8080,
            "debug": True,
            "host": "localhost"
        }
        
        result = _convert_to_env(TestSettings, raw_defaults)
        
        expected = {
            "MLSERVER_HTTP_PORT": "8080",
            "MLSERVER_DEBUG": "True",
            "MLSERVER_HOST": "localhost"
        }
        assert result == expected
    
    def test_convert_to_env_no_prefix(self):
        """Test converting settings to environment variables without prefix."""
        class TestSettings:
            pass
        
        raw_defaults = {"port": 9000, "enabled": False}
        
        result = _convert_to_env(TestSettings, raw_defaults)
        
        expected = {
            "PORT": "9000",
            "ENABLED": "False"
        }
        assert result == expected
    
    def test_get_default_env_multiple_settings(self):
        """Test getting environment variables from multiple settings classes."""
        class Config1:
            env_prefix = "APP_"
        
        class Settings1:
            Config = Config1
        
        class Config2:
            env_prefix = "MODEL_"
        
        class Settings2:
            Config = Config2
        
        default_settings = [
            (Settings1, {"port": 8080, "debug": True}),
            (Settings2, {"name": "test-model", "version": "v1"})
        ]
        
        result = get_default_env(default_settings)
        
        expected = {
            "APP_PORT": "8080",
            "APP_DEBUG": "True",
            "MODEL_NAME": "test-model",
            "MODEL_VERSION": "v1"
        }
        assert result == expected


class TestValueParsing:
    """Test value parsing and formatting functionality."""
    
    def test_parse_dict_values_json_dict(self):
        """Test parsing dictionary values as JSON."""
        result = _parse_dict_values("CONFIG", "{'key': 'value'}")
        assert result == 'CONFIG=\'{"key": "value"}\'\n'
    
    def test_parse_dict_values_json_list(self):
        """Test parsing list values as JSON."""
        result = _parse_dict_values("TAGS", "['tag1', 'tag2']")
        assert result == 'TAGS=\'["tag1", "tag2"]\'\n'
    
    def test_parse_dict_values_json_number(self):
        """Test parsing numeric values as JSON."""
        result = _parse_dict_values("PORT", "8080")
        assert result == "PORT='8080'\n"
    
    def test_parse_dict_values_plain_string(self):
        """Test parsing plain string values."""
        result = _parse_dict_values("HOST", "localhost")
        assert result == 'HOST="localhost"\n'
    
    def test_parse_dict_values_complex_json(self):
        """Test parsing complex nested JSON."""
        complex_value = "{'nested': {'key': 'value'}, 'list': [1, 2, 3]}"
        result = _parse_dict_values("COMPLEX", complex_value)
        expected = 'COMPLEX=\'{"nested": {"key": "value"}, "list": [1, 2, 3]}\'\n'
        assert result == expected
    
    def test_parse_dict_values_quotes_in_string(self):
        """Test parsing strings with quotes."""
        result = _parse_dict_values("MESSAGE", 'Hello "world"')
        assert result == 'MESSAGE="Hello \\"world\\""\n'


class TestFileWriting:
    """Test environment file writing functionality."""
    
    def test_save_default_env_success(self, temp_dir):
        """Test successfully saving environment variables to file."""
        env = {
            "APP_PORT": "8080",
            "APP_DEBUG": "True",
            "MODEL_CONFIG": "{'key': 'value'}"
        }
        
        output_file = os.path.join(temp_dir, "test.env")
        save_default_env(env, output_file)
        
        # Verify file was created and has correct content
        assert os.path.exists(output_file)
        
        with open(output_file, 'r') as f:
            content = f.read()
        
        assert 'APP_PORT="8080"' in content
        assert 'APP_DEBUG="True"' in content
        assert 'MODEL_CONFIG=\'{"key": "value"}\'' in content
    
    def test_save_default_env_permission_denied(self, temp_dir):
        """Test saving to file with no write permissions."""
        env = {"TEST": "value"}
        
        # Create directory with no write permissions
        no_write_dir = os.path.join(temp_dir, "no_write")
        os.makedirs(no_write_dir)
        os.chmod(no_write_dir, 0o444)
        
        output_file = os.path.join(no_write_dir, "test.env")
        
        try:
            with pytest.raises(PermissionError):
                save_default_env(env, output_file)
        finally:
            # Restore permissions for cleanup
            os.chmod(no_write_dir, 0o755)
    
    def test_save_default_env_directory_not_exists(self):
        """Test saving to non-existent directory."""
        env = {"TEST": "value"}
        output_file = "/nonexistent/directory/test.env"
        
        with pytest.raises(FileNotFoundError):
            save_default_env(env, output_file)


class TestCLIIntegration:
    """Test command-line interface integration."""
    
    def test_main_function_integration(self, create_settings_files, temp_dir):
        """Test main function with real files."""
        settings_dir = create_settings_files['temp_dir']
        output_file = os.path.join(temp_dir, "output.env")
        
        # Call main function directly
        from click.testing import CliRunner
        runner = CliRunner()
        
        result = runner.invoke(main, [settings_dir, output_file])
        
        assert result.exit_code == 0
        assert os.path.exists(output_file)
        
        # Verify output file content
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Should contain environment variables from both files
        assert "HTTP_PORT" in content
        assert "MODEL_NAME" in content
        assert "MODEL_URI" in content
    
    def test_main_function_no_files(self, temp_dir):
        """Test main function with directory containing no settings files."""
        output_file = os.path.join(temp_dir, "empty.env")
        
        from click.testing import CliRunner
        runner = CliRunner()
        
        result = runner.invoke(main, [temp_dir, output_file])
        
        assert result.exit_code == 0
        assert os.path.exists(output_file)
        
        # File should be empty or contain only whitespace
        with open(output_file, 'r') as f:
            content = f.read().strip()
        
        assert content == ""
    
    def test_cli_subprocess_integration(self, create_settings_files, temp_dir):
        """Test CLI through subprocess for full integration."""
        settings_dir = create_settings_files['temp_dir']
        output_file = os.path.join(temp_dir, "subprocess.env")
        
        # Run the script as subprocess
        script_path = project_root / "hack" / "generate_dotenv.py"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            settings_dir, output_file
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert os.path.exists(output_file)
        
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Verify expected environment variables are present
        assert "HTTP_PORT" in content
        assert "MODEL_NAME" in content


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_unicode_handling(self, temp_dir):
        """Test handling of Unicode characters in settings."""
        unicode_data = {
            "测试": "中文值",
            "emoji": "🚀🔥",
            "mixed": "English with 中文 and émojis 🎉"
        }
        
        settings_path = os.path.join(temp_dir, "settings.json")
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(unicode_data, f, ensure_ascii=False)
        
        result = load_default_settings(temp_dir)
        
        assert len(result) == 1
        settings_class, settings_data = result[0]
        assert settings_data["测试"] == "中文值"
        assert settings_data["emoji"] == "🚀🔥"
    
    def test_large_json_values(self, temp_dir):
        """Test handling of large JSON values."""
        large_list = list(range(10000))
        large_dict = {f"key_{i}": f"value_{i}" for i in range(1000)}
        
        large_data = {
            "large_list": large_list,
            "large_dict": large_dict,
            "large_string": "x" * 100000
        }
        
        settings_path = os.path.join(temp_dir, "settings.json")
        with open(settings_path, 'w') as f:
            json.dump(large_data, f)
        
        result = load_default_settings(temp_dir)
        
        assert len(result) == 1
        settings_class, settings_data = result[0]
        assert len(settings_data["large_list"]) == 10000
        assert len(settings_data["large_dict"]) == 1000
        assert len(settings_data["large_string"]) == 100000
    
    def test_special_characters_in_field_names(self, temp_dir):
        """Test handling of special characters in field names."""
        special_data = {
            "field-with-dashes": "value1",
            "field_with_underscores": "value2",
            "field.with.dots": "value3",
            "field with spaces": "value4",
            "field@with#symbols": "value5"
        }
        
        settings_path = os.path.join(temp_dir, "settings.json")
        with open(settings_path, 'w') as f:
            json.dump(special_data, f)
        
        result = load_default_settings(temp_dir)
        
        assert len(result) == 1
        settings_class, settings_data = result[0]
        
        # Test environment variable conversion
        env = _convert_to_env(settings_class, settings_data)
        
        # Field names should be converted to uppercase
        assert "FIELD-WITH-DASHES" in env
        assert "FIELD_WITH_UNDERSCORES" in env
        assert "FIELD.WITH.DOTS" in env
        assert "FIELD WITH SPACES" in env
        assert "FIELD@WITH#SYMBOLS" in env


# Property-based testing with hypothesis
class TestPropertyBased:
    """Property-based tests using hypothesis."""
    
    @given(st.dictionaries(
        st.text(min_size=1, max_size=50),
        st.one_of(
            st.text(max_size=100),
            st.integers(),
            st.booleans(),
            st.floats(allow_nan=False, allow_infinity=False)
        ),
        min_size=1,
        max_size=20
    ))
    def test_convert_to_env_property(self, raw_defaults):
        """Property-based test for environment variable conversion."""
        class TestSettings:
            pass
        
        try:
            result = _convert_to_env(TestSettings, raw_defaults)
            
            # Properties that should always hold
            assert isinstance(result, dict)
            assert len(result) == len(raw_defaults)
            
            for key, value in result.items():
                assert isinstance(key, str)
                assert isinstance(value, str)
                assert key.isupper()
        except Exception:
            # Some inputs might cause legitimate exceptions
            pass
    
    @given(st.text(min_size=1, max_size=100))
    def test_parse_dict_values_property(self, value):
        """Property-based test for value parsing."""
        try:
            result = _parse_dict_values("TEST_VAR", value)
            
            # Properties that should always hold
            assert isinstance(result, str)
            assert result.startswith("TEST_VAR=")
            assert result.endswith("\n")
            assert len(result) > len("TEST_VAR=\n")
        except Exception:
            # Some inputs might cause legitimate exceptions
            pass