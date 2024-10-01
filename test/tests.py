import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.model import EarthquakeModel
from main import load_model
from src.prediction.inference import (
    input_size, 
    hidden_size,
    num_layers,
    output_size,
    dropout_prob
)

import unittest
import torch

class TestLoadModel(unittest.TestCase):
    def setUp(self):
        # Set up any necessary variables or paths
        self.model_path = r"C:\Projs\COde\Earthquake\eq_prediction\src\model\earthquake_best_model.pth"
        self.device = False

    def test_load_existing_model(self):
        # Test loading an existing model
        model, success = load_model(self.model_path)
        
        self.assertTrue(success, "Model loading should succeed")
        self.assertIsInstance(model, EarthquakeModel, "Loaded model should be an instance of EarthquakeModel")
        self.assertEqual(next(model.parameters()).is_cuda, self.device, "Model should be on the correct device")

    def test_load_nonexistent_model(self):
        # Test loading a nonexistent model
        nonexistent_path = r"C:\Projs\COde\Earthquake\eq_prediction\src\model\earthquake_best_model.pth"
        model, success = load_model(nonexistent_path)
        self.assertFalse(success, "Model loading should fail for nonexistent path")
        self.assertIsNone(model, "Model should be None when loading fails")

if __name__ == "__main__":
    unittest.main()
