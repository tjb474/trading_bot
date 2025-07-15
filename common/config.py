import os
from typing import Dict, List, Any, Optional
import yaml
from pathlib import Path
from dotenv import load_dotenv

class Config:
    """Configuration manager that combines settings from config.yaml and environment variables."""
    
    def __init__(self, config_file: str = 'config.yaml'):
        self.base_dir = Path(__file__).resolve().parent.parent
        self._load_environment()
        self._load_yaml(config_file)

    def _load_environment(self) -> None:
        """Load environment variables from .env file"""
        env_path = self.base_dir / '.env'
        load_dotenv(dotenv_path=env_path)
        
    def _load_yaml(self, config_file: str) -> None:
        """Load and validate the YAML configuration file"""
        yaml_path = self.base_dir / config_file
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
            
        with open(yaml_path, 'r') as f:
            self._config = yaml.safe_load(f)

    @property
    def active_strategy(self) -> str:
        """Name of the currently active trading strategy"""
        return self._config['active_strategy']

    @property
    def trading_params(self) -> Dict[str, Any]:
        """Global trading parameters like initial cash and commission"""
        return self._config['trading']

    @property
    def data_path(self) -> Path:
        """Absolute path to the data file"""
        return self.base_dir / self._config['paths']['data']

    def get_strategy_config(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for a specific strategy
        
        Args:
            strategy_name: Name of the strategy. If None, uses active_strategy
            
        Returns:
            Dictionary containing all configuration for the strategy
        """
        name = strategy_name or self.active_strategy
        try:
            return self._config['strategies'][name]
        except KeyError:
            raise ValueError(f"Strategy '{name}' not found in config")

    def get_model_path(self, strategy_name: Optional[str] = None) -> Path:
        """
        Get the absolute path to a strategy's model file
        
        Args:
            strategy_name: Name of the strategy. If None, uses active_strategy
            
        Returns:
            Absolute Path to the model file
        """
        config = self.get_strategy_config(strategy_name)
        return self.base_dir / config['model']['path']

# Global configuration instance
config = Config()