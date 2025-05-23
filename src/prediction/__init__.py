"""
Prediction module for making earthquake forecasts.
Contains functions for model inference and future predictions.
"""

from .inference import (
    test_step,
    load_model,
    future_forecast,
    generateDateRange,
    input_size,
    hidden_size,
    num_layers,
    output_size,
    dropout_prob
)

__all__ = [
    'test_step',
    'load_model',
    'future_forecast',
    'generateDateRange',
    'input_size',
    'hidden_size',
    'num_layers', 
    'output_size',
    'dropout_prob'
]