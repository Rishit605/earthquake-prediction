import numpy as np

from sklearn.impute import KNNImputer

class CustomKNN:
    def __init__(self, neighbors=5) -> None:
        self.n_nei = neighbors

    def calc_euclid_dist(self, x1, x2):
        return np.linalg.norm(x1-x2)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    
    def predict(self, X):
        """
        Take the dataset and predicts labels for each example.
        """

        predictions = []

        for x in X:
            prediction = self._predict(x)
            predictions.append(prediction)
        return np.array(predictions)

    def _predict(self, x):
        distances = []

        for X in self.X_train:
            distance = self.calc_euclid_dist(x, X)
            distances.append(distance)

        distances = np.array(distances)

        n_idxs = np.argsort(distances)[:self.n_nei]

                # Get labels of n-neighbour indexes
        labels = self.y_train[n_idxs]                  
        labels = list(labels)
        # Get the most frequent class in the array
        most_occuring_value = max(labels, key=labels.count)
        return most_occuring_value

if __name__ == "__main__":
    pass