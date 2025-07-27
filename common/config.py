import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

class Config:
    """
    A unified configuration class that loads settings from config.yaml and .env files.
    It provides a single, reliable source of truth for all configuration needs.
    """
    def __init__(self, config_file: str = 'config.yaml'):
        # 1. Set up the base directory
        # This is the root of your project (the 'trading_bot' folder's parent)
        self.BASE_DIR = Path(__file__).resolve().parent.parent

        # 2. Load environment variables from the .env file in the root directory
        dotenv_path = self.BASE_DIR / '.env'
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path)
        else:
            print("Warning: .env file not found. Secrets will not be loaded.")

        # 3. Load the YAML configuration file
        yaml_path = self.BASE_DIR / config_file
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Config file not found at '{yaml_path}'")
        with open(yaml_path, 'r') as f:
            self.yaml_data = yaml.safe_load(f)

    # --- Properties for Tunable General Parameters (from yaml) ---
    @property
    def general(self) -> dict:
        return self.yaml_data['general']

    @property
    def active_strategy(self) -> str:
        return self.yaml_data['active_strategy']
    
    @property
    def data_path(self) -> str:
        """Get the data file path from the paths section."""
        return self.yaml_data['paths']['data']
    
    @property
    def trading_params(self) -> dict:
        """Get the trading parameters section."""
        return self.yaml_data['trading']
    
    # --- Methods for strategy-specific parameters (from yaml) ---
    def get_strategy_config(self, strategy_name: str = None) -> dict:
        """
        Returns the configuration dictionary for a given strategy.
        If no name is provided, it uses the active_strategy from the config.
        """
        if strategy_name is None:
            strategy_name = self.active_strategy
            
        try:
            return self.yaml_data['strategies'][strategy_name]
        except KeyError:
            raise ValueError(f"Configuration for strategy '{strategy_name}' not found in config.yaml.")
    
    def get_strategy_params(self, strategy_name: str = None) -> dict:
        """
        Returns the parameter dictionary for a given strategy.
        If no name is provided, it uses the active_strategy from the config.
        """
        if strategy_name is None:
            strategy_name = self.active_strategy
            
        try:
            return self.yaml_data['strategies'][strategy_name]
        except KeyError:
            raise ValueError(f"Parameters for strategy '{strategy_name}' not found in config.yaml.")

    # --- Properties for Dynamic Paths (combined logic) ---
    @property
    def DATA_DIR(self) -> Path:
        return self.BASE_DIR / 'data'
        
    @property
    def DATA_FILE_PATH(self) -> Path:
        """Gets the data file path from YAML and makes it an absolute path."""
        relative_path = Path(self.general['data_file_path'])
        return self.BASE_DIR / relative_path

    def get_model_path(self, strategy_name: str = None) -> Path:
        """Gets the model file path for a strategy and makes it absolute."""
        params = self.get_strategy_params(strategy_name)
        # Check if model path is under 'model' section (new structure) or directly (old structure)
        if 'model' in params and 'path' in params['model']:
            relative_path = Path(params['model']['path'])
        elif 'model_file_path' in params:
            relative_path = Path(params['model_file_path'])
        else:
            raise KeyError(f"Model path not found for strategy '{strategy_name}'. Expected 'model.path' or 'model_file_path' in config.")
        return self.BASE_DIR / relative_path


# --- Global Instance ---
# Any module in your project can now just 'from common.config import config'
# to get access to the fully loaded and processed configuration.
config = Config()