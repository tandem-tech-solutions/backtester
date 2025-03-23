import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages loading and saving of strategy configurations"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager
        
        Args:
            config_path: Path to the configuration file, None uses default config.json
        """
        self.config_file = config_path if config_path else "config.json"
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Load configuration from file or create default if doesn't exist
        
        Returns:
            Dictionary containing strategy configurations
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                return config
            except json.JSONDecodeError:
                logger.error(f"Error decoding {self.config_file}. Using default configuration.")
                return self._create_default_config()
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}. Using default configuration.")
                return self._create_default_config()
        else:
            logger.info(f"Configuration file {self.config_file} not found. Creating default configuration.")
            default_config = self._create_default_config()
            self._save_config(default_config)
            return default_config
            
    def _create_default_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Create default configuration with pre-defined strategies
        
        Returns:
            Dictionary containing default strategy configurations
        """
        default_config = {
            "MA_CROSSOVER": {
                "short_window": 20,
                "long_window": 50,
                "warmup_period": 15,
                "description": "Moving Average Crossover Strategy"
            },
            "INSIDE_CANDLE_RSI": {
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "warmup_period": 20,
                "stop_loss_pct": 2.0,
                "take_profit_pct": 4.0,
                "trailing_stop_pct": 1.5,
                "max_bars": None,
                "description": "Inside Candle Breakout with RSI Confirmation Strategy"
            }
        }
        return default_config
    
    def _save_config(self, config: Dict[str, Dict[str, Any]]) -> None:
        """
        Save configuration to file
        
        Args:
            config: Configuration dictionary to save
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def get_strategy_config(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific strategy
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary containing strategy parameters or None if not found
        """
        if strategy_name in self.config:
            return self.config[strategy_name]
        else:
            logger.warning(f"Strategy '{strategy_name}' not found in configuration")
            return None
    
    def get_strategy_params(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific strategy formatted for environment variables
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary containing strategy parameters with proper names for env vars
        """
        strategy_config = self.get_strategy_config(strategy_name)
        if not strategy_config:
            return {}
            
        # Format parameters as environment variables
        env_params = {}
        for param, value in strategy_config.items():
            # Skip description field
            if param == "description":
                continue
                
            # Keep strategy_type as is, don't convert to uppercase
            if param == "strategy_type":
                env_params[param] = value
                continue
                
            # Convert parameter names to uppercase
            env_name = param.upper()
            
            # Special case for warmup_period parameter for the inside candle strategy
            if strategy_name == "INSIDE_CANDLE_RSI" and param == "warmup_period":
                env_name = "IC_WARMUP_PERIOD"
                
            # Skip None values
            if value is not None:
                env_params[env_name] = value
                
        return env_params
    
    def save_strategy_config(self, strategy_name: str, params: Dict[str, Any]) -> None:
        """
        Save or update configuration for a specific strategy
        
        Args:
            strategy_name: Name of the strategy
            params: Strategy parameters
        """
        self.config[strategy_name] = params
        self._save_config(self.config)
        logger.info(f"Updated configuration for strategy '{strategy_name}'")
    
    def get_available_strategies(self) -> Dict[str, str]:
        """
        Get list of available strategies with descriptions
        
        Returns:
            Dictionary with strategy names as keys and descriptions as values
        """
        return {name: config.get("description", "No description available") 
                for name, config in self.config.items()} 