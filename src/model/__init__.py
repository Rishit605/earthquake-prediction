"""
Model module containing the neural network architecture for earthquake prediction.
"""

from .model import EarthquakeModel, ModelCheckPoint, Early_Stopping

__all__ = ['EarthquakeModel', 'ModelCheckPoint', 'Early_Stopping', ]