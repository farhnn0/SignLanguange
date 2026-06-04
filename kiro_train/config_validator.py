"""
Configuration Validation Module for Augmentation Pipeline

This module provides validation utilities for the augmentation configuration file.
It validates types, ranges, and consistency of all configuration parameters.
"""

import os
import yaml
import logging
from typing import Dict, Any, Tuple, List


def get_default_config() -> Dict[str, Any]:
    """
    Returns hardcoded default configuration when config file is missing.
    
    Returns:
        Dict[str, Any]: Default configuration dictionary with all required fields
    """
    return {
        # Global settings
        "random_seed": 42,
        "batch_size": 32,
        "epochs": 50,
        "augmentation_multiplier": 3,
        "augmentation_mode": "landmark",
        
        # Horizontal flip
        "enable_horizontal_flip": True,
        "flip_probability": 0.5,
        
        # Speed variation
        "enable_speed_variation": True,
        "speed_factors": [0.8, 0.9, 1.0, 1.1, 1.2],
        "speed_probability": 0.7,
        
        # Brightness & contrast
        "enable_brightness_contrast": True,
        "brightness_range": [-0.2, 0.2],
        "contrast_range": [0.8, 1.2],
        "brightness_contrast_probability": 0.5,
        
        # Random crop & resize
        "enable_random_crop_resize": True,
        "scale_range": [0.85, 1.15],
        "crop_probability": 0.5,
        
        # Gaussian noise
        "enable_gaussian_noise": True,
        "landmark_noise_std": 0.005,
        "pixel_noise_std": 5.0,
        "noise_probability": 0.5,
        
        # Rotation
        "enable_rotation": True,
        "rotation_range_degrees": [-10.0, 10.0],
        "rotation_probability": 0.5,
    }


def validate_config(config_path: str = None, logger: logging.Logger = None) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validates augmentation configuration from file or uses defaults.
    
    This function:
    1. Attempts to load configuration from YAML file
    2. Falls back to hardcoded defaults if file not found (with warning)
    3. Validates all field types and value ranges
    4. Returns validated config and list of any warnings/errors
    
    Args:
        config_path (str, optional): Path to augmentation_config.yaml file.
                                     If None, looks for 'augmentation_config.yaml' in current dir.
        logger (logging.Logger, optional): Logger instance for logging messages.
                                          If None, uses print statements.
    
    Returns:
        Tuple[Dict[str, Any], List[str]]: 
            - Validated configuration dictionary
            - List of validation messages (warnings, errors)
    
    Raises:
        ValueError: If configuration has invalid types or value ranges
    """
    
    messages = []
    
    # Determine config file path
    if config_path is None:
        config_path = "augmentation_config.yaml"
    
    config = None
    
    # Try to load config file
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            if logger:
                logger.info(f"Loaded configuration from {config_path}")
            else:
                print(f"[INFO] Loaded configuration from {config_path}")
        except Exception as e:
            error_msg = f"Error reading config file {config_path}: {str(e)}"
            if logger:
                logger.error(error_msg)
            else:
                print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)
    else:
        # Config file not found, use defaults
        warning_msg = f"Config file not found at {config_path}. Using hardcoded defaults."
        messages.append(warning_msg)
        if logger:
            logger.warning(warning_msg)
        else:
            print(f"[WARNING] {warning_msg}")
        config = get_default_config()
    
    # Validate configuration
    validation_errors = _validate_config_fields(config)
    
    if validation_errors:
        error_details = "\n".join(validation_errors)
        full_error = f"Configuration validation failed:\n{error_details}"
        if logger:
            logger.error(full_error)
        else:
            print(f"[ERROR] {full_error}")
        raise ValueError(full_error)
    
    # Validation successful
    if logger:
        logger.info("Configuration validation passed")
    else:
        print("[INFO] Configuration validation passed")
    
    return config, messages


def _validate_config_fields(config: Dict[str, Any]) -> List[str]:
    """
    Internal function to validate all configuration fields.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to validate
    
    Returns:
        List[str]: List of validation error messages (empty if all valid)
    """
    
    errors = []
    
    # Define validation rules for each field
    validation_rules = {
        # Global settings
        "random_seed": {
            "type": int,
            "min": 0,
            "max": 2**31 - 1,
            "required": True
        },
        "batch_size": {
            "type": int,
            "min": 1,
            "max": 1024,
            "required": True
        },
        "epochs": {
            "type": int,
            "min": 1,
            "max": 1000,
            "required": True
        },
        "augmentation_multiplier": {
            "type": int,
            "min": 1,
            "max": 100,
            "required": True
        },
        "augmentation_mode": {
            "type": str,
            "allowed_values": ["landmark", "video_prepass"],
            "required": True
        },
        
        # Horizontal flip
        "enable_horizontal_flip": {
            "type": bool,
            "required": True
        },
        "flip_probability": {
            "type": (int, float),
            "min": 0.0,
            "max": 1.0,
            "required": True
        },
        
        # Speed variation
        "enable_speed_variation": {
            "type": bool,
            "required": True
        },
        "speed_factors": {
            "type": list,
            "element_type": (int, float),
            "element_min": 0.5,
            "element_max": 2.0,
            "required": True
        },
        "speed_probability": {
            "type": (int, float),
            "min": 0.0,
            "max": 1.0,
            "required": True
        },
        
        # Brightness & contrast
        "enable_brightness_contrast": {
            "type": bool,
            "required": True
        },
        "brightness_range": {
            "type": (list, tuple),
            "length": 2,
            "element_type": (int, float),
            "required": True
        },
        "contrast_range": {
            "type": (list, tuple),
            "length": 2,
            "element_type": (int, float),
            "element_min": 0.5,
            "element_max": 2.0,
            "required": True
        },
        "brightness_contrast_probability": {
            "type": (int, float),
            "min": 0.0,
            "max": 1.0,
            "required": True
        },
        
        # Random crop & resize
        "enable_random_crop_resize": {
            "type": bool,
            "required": True
        },
        "scale_range": {
            "type": (list, tuple),
            "length": 2,
            "element_type": (int, float),
            "element_min": 0.5,
            "element_max": 2.0,
            "required": True
        },
        "crop_probability": {
            "type": (int, float),
            "min": 0.0,
            "max": 1.0,
            "required": True
        },
        
        # Gaussian noise
        "enable_gaussian_noise": {
            "type": bool,
            "required": True
        },
        "landmark_noise_std": {
            "type": (int, float),
            "min": 0.0,
            "max": 0.1,
            "required": True
        },
        "pixel_noise_std": {
            "type": (int, float),
            "min": 0.0,
            "max": 50.0,
            "required": True
        },
        "noise_probability": {
            "type": (int, float),
            "min": 0.0,
            "max": 1.0,
            "required": True
        },
        
        # Rotation
        "enable_rotation": {
            "type": bool,
            "required": True
        },
        "rotation_range_degrees": {
            "type": (list, tuple),
            "length": 2,
            "element_type": (int, float),
            "element_min": -180.0,
            "element_max": 180.0,
            "required": True
        },
        "rotation_probability": {
            "type": (int, float),
            "min": 0.0,
            "max": 1.0,
            "required": True
        },
    }
    
    # Check each required field
    for field_name, rules in validation_rules.items():
        if rules.get("required", False):
            if field_name not in config:
                errors.append(f"Required field '{field_name}' is missing")
                continue
            
            field_value = config[field_name]
            
            # Check type
            expected_type = rules.get("type")
            if expected_type and not isinstance(field_value, expected_type):
                errors.append(
                    f"Field '{field_name}': expected type {expected_type}, "
                    f"got {type(field_value).__name__}"
                )
                continue
            
            # Check if value is in allowed values (for strings)
            if "allowed_values" in rules:
                if field_value not in rules["allowed_values"]:
                    errors.append(
                        f"Field '{field_name}': value '{field_value}' not in "
                        f"allowed values {rules['allowed_values']}"
                    )
                continue
            
            # Check range (for numeric values)
            if "min" in rules and field_value < rules["min"]:
                errors.append(
                    f"Field '{field_name}': value {field_value} is below minimum {rules['min']}"
                )
            if "max" in rules and field_value > rules["max"]:
                errors.append(
                    f"Field '{field_name}': value {field_value} is above maximum {rules['max']}"
                )
            
            # Check list/tuple length
            if rules.get("type") in (list, tuple, [list, tuple]) or isinstance(rules.get("type"), (list, tuple)):
                if "length" in rules:
                    if len(field_value) != rules["length"]:
                        errors.append(
                            f"Field '{field_name}': expected length {rules['length']}, "
                            f"got {len(field_value)}"
                        )
                    
                    # Check element types and ranges
                    element_type = rules.get("element_type")
                    for i, element in enumerate(field_value):
                        if element_type and not isinstance(element, element_type):
                            errors.append(
                                f"Field '{field_name}[{i}]': expected type {element_type}, "
                                f"got {type(element).__name__}"
                            )
                            # Skip range checks if type is wrong
                            continue
                        
                        if "element_min" in rules and element < rules["element_min"]:
                            errors.append(
                                f"Field '{field_name}[{i}]': value {element} is below "
                                f"minimum {rules['element_min']}"
                            )
                        if "element_max" in rules and element > rules["element_max"]:
                            errors.append(
                                f"Field '{field_name}[{i}]': value {element} is above "
                                f"maximum {rules['element_max']}"
                            )
            
            # Check list elements (variable length)
            if rules.get("element_type") and "length" not in rules:
                element_type = rules.get("element_type")
                for i, element in enumerate(field_value):
                    if not isinstance(element, element_type):
                        errors.append(
                            f"Field '{field_name}[{i}]': expected type {element_type}, "
                            f"got {type(element).__name__}"
                        )
                        # Skip range checks if type is wrong
                        continue
                    
                    if "element_min" in rules and element < rules["element_min"]:
                        errors.append(
                            f"Field '{field_name}[{i}]': value {element} is below "
                            f"minimum {rules['element_min']}"
                        )
                    if "element_max" in rules and element > rules["element_max"]:
                        errors.append(
                            f"Field '{field_name}[{i}]': value {element} is above "
                            f"maximum {rules['element_max']}"
                        )
    
    return errors


def load_and_validate_config(config_path: str = None, logger: logging.Logger = None) -> Dict[str, Any]:
    """
    Convenience function to load and validate config in one call.
    
    Args:
        config_path (str, optional): Path to configuration file
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        Dict[str, Any]: Validated configuration dictionary
    
    Raises:
        ValueError: If configuration is invalid
    """
    config, messages = validate_config(config_path, logger)
    return config
