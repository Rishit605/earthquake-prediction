"""
Grid-search evaluation for the scratch Decision Tree regressor (DTRegressor pipeline).

Run from project root:
    python test/Dtreg.py
"""

import concurrent.futures
import itertools
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_new import DTRegressor
from src.model.decision_tree_scratch import DecisionTreeR
from src.helpers.utils import r2_Loss

# Module-level data for worker processes (Windows multiprocessing)
X_train = None
y_train = None
X_valid = None
y_valid = None

def evaluate_tree(params):
    depth, sample = params
    dtr = DTRegressor()
    dtr.fit(X_train, y_train)
    score = dtr._evaluate_model(max_depth=depth, min_samples_split=sample)
    return depth, sample, score


def load_data():
    reg = DTRegressor()
    train_x, train_y = reg.data_prep()
    valid_x, valid_y = reg.data_prep(valid=True)
    return train_x, train_y, valid_x, valid_y


if __name__ == "__main__":
    X_train, y_train, X_valid, y_valid = load_data()
    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}")

    # All parameter combinations: (2, 2), (2, 3), ... (20, 20)
    param_grid = list(itertools.product(range(2, 21), range(2, 21)))
    print(f"Evaluating {len(param_grid)} depth × min_sample combinations...")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(evaluate_tree, param_grid)

        best_score = float("-inf")
        best_params = None

        for depth, sample, score in results:
            print(f"Depth|Sample: {depth}|{sample} --> Model Score: {score}")
            if score > best_score:
                best_score = score
                best_params = (depth, sample)

    if best_params is not None:
        print(
            f"\nBest: depth={best_params[0]}, min_samples_split={best_params[1]}, "
            f"R²={best_score:.6f}"
        )
