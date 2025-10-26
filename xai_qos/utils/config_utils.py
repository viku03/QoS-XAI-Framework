# Configuration Manager and Utilities
"""
Configuration management and utility functions for XAI-QoS Framework
"""

import yaml
import logging
import os
from typing import Dict, Any
from pathlib import Path

class ConfigManager:
    """Manages configuration loading and validation"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            return ConfigManager.get_default_config()
        except Exception as e:
            logging.error(f"Error loading config from {config_path}: {e}")
            return ConfigManager.get_default_config()
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'device': 'cpu',
            'classifier': {
                'input_dim': 50,
                'hidden_dim': 128,
                'n_heads': 8,
                'n_layers': 3,
                'n_classes': 10,
                'dropout': 0.1,
                'model_path': None,
                'class_names': [
                    'youtube', 'netflix', 'zoom', 'teams', 'ftp', 
                    'http_browsing', 'gaming', 'voip', 'streaming', 'background'
                ]
            },
            'explainer': {
                'importance_threshold': 0.05,
                'top_k_features': 5
            },
            'policy_translator': {
                'enable_dynamic_adjustment': True,
                'congestion_threshold': 0.8
            },
            'sdn_controller': {
                'monitoring_interval': 5,
                'flow_timeout': 300,
                'controller_port': 6633
            },
            'qos_policies': {
                'enable_preemption': True,
                'max_queues': 8,
                'default_queue': 3
            }
        }

def setup_logging(level: str = 'INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('xai_qos.log')
        ]
    )