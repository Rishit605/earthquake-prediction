"""
Model module containing the neural network architecture for earthquake prediction.
"""

from .model import EarthquakeModel, ModelCheckPoint, Early_Stopping
from .lr_scratch import LinearR
__all__ = ['EarthquakeModel', 'ModelCheckPoint', 'Early_Stopping', 'LinearR']