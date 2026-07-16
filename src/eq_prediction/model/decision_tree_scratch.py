## TODO: Correctly implement the regression logic

import numpy as np

class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        gain=None,
        value=None
    ) -> None:
    
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value
 

class DecisionTreeC:

    def __init__(self, max_depth=2, min_sample=2):
        self.min_sample = min_sample
        self.max_depth = max_depth

    def split(self, data, feature, threshold):
        left = []
        right = []
   


        for d in data:
            if d[feature] > threshold:
                right.append(d)
            else:
                left.append(d)

        left = np.array(left)
        right = np.array(right)

    def entropy(self, act):
        ent = 0

        labels = np.unique(act)

        for l in labels:
            exm = act[act == l]
            p1 = len(exm) / len(act)
            ent += -p1 * np.log(p1)

        return ent

    def info_gain(self, parent, left, right):
        parent_ent = self.entropy(parent)
        
        l_w = len(left) / len(parent)
        r_w = len(right) / len(parent)
        
        left_ent = self.entropy(left)
        right_ent = self.entropy(right)

        gain = parent_ent - ((l_w * left_ent) + (r_w * right_ent))
        return gain

    def best_split(self, dataset, num_samples, num_features):
        best_split = {'gain':- 1, 'feature': None, 'threshold': None}
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:    
                left_dataset, right_dataset = self.split_data(dataset, feature_index, threshold)    
                if len(left_dataset) and len(right_dataset):        
                    y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]        
                    information_gain = self.information_gain(y, left_y, right_y)        
                    if information_gain > best_split["gain"]:
                        best_split["feature"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["left_dataset"] = left_dataset
                        best_split["right_dataset"] = right_dataset
                        best_split["gain"] = information_gain
        return best_split


    def calculate_leaf_value(self,act):
        act = list(act)
        most_occuring_value = max(act, key=act.count)
        return most_occuring_value


    def _calculate_mse(self, y):
        if len(y) == 0:
            return 0
        mean_y = np.mean(y)
        return np.mean(np.square(y - mean_y))

        
    def build_tree(self, dataset, current_depth=0):
        """
        Recursively builds a decision tree from the given dataset.

        Args:
        dataset (ndarray): The dataset to build the tree from.
        current_depth (int): The current depth of the tree.

        Returns:
        Node: The root node of the built decision tree.
        """
        # split the dataset into X, y values
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape
        # keeps spliting until stopping conditions are met
        if n_samples >= self.min_samples and current_depth <= self.max_depth:
            # Get the best split
            best_split = self.best_split(dataset, n_samples, n_features)
            # Check if gain isn't zero
            if best_split["gain"]:
                # continue splitting the left and the right child. Increment current depth
                left_node = self.build_tree(best_split["left_dataset"], current_depth + 1)
                right_node = self.build_tree(best_split["right_dataset"], current_depth + 1)
                # return decision node
                return Node(best_split["feature"], best_split["threshold"],
                            left_node, right_node, best_split["gain"])

        # compute leaf node value
        leaf_value = self.calculate_leaf_value(y)
        # return leaf node value
        return Node(value=leaf_value)
    
    def fit(self, X, y):
        dataset = np.concatenate((X, y), axis=1)  
        self.root = self.build_tree(dataset)

    def predict(self, X):
        # Create an empty list to store the predictions
        predictions = []
        # For each instance in X, make a prediction by traversing the tree
        for x in X:
            prediction = self.make_prediction(x, self.root)
            # Append the prediction to the list of predictions
            predictions.append(prediction)
        # Convert the list to a numpy array and return it
        np.array(predictions)
        return predictions
    
    def make_prediction(self, x, node):
        # if the node has value i.e it's a leaf node extract it's value
        if node.value != None: 
            return node.value
        else:
            #if it's node a leaf node we'll get it's feature and traverse through the tree accordingly
            feature = x[node.feature]
            if feature <= node.threshold:
                return self.make_prediction(x, node.left)
            else:
                return self.make_prediction(x, node.right)


class DecisionTreeR:
    def __init__(self, max_depth=3, min_sample=2):
        self.max_depth = max_depth
        self.min_sample = min_sample
        self.root = None
    
    def _calc_mse(self, y):
        if len(y) == 0:
            return 0
        mean_y = np.mean(y)
        return np.mean(np.square(y - mean_y))

    def _best_split(self, X, y):
        best_mse = -1
        best_feat, best_thresh = None, None
        best_X_left, best_y_left = None, None
        best_X_right, best_y_right = None, None
        
        # Calculate the baseline MSE of the parent node before splitting
        parent_mse = self._calc_mse(y)
        n_samples, n_features = X.shape

        for f in range(n_features):
            X_cols = X[:, f]
            
            # Sampling the threshold values due to continuous nature of certain columns
            perp = np.percentile(X_cols, np.linspace(10,90,20))
            thresh = (perp[:-1] + perp[1:]) / 2

            for t in thresh:
                left_m = X_cols <= t
                right_m = X_cols > t

                if np.sum(left_m) == 0 or np.sum(right_m) == 0:
                    continue

                y_left, y_right = y[left_m], y[right_m]

                mse_left = self._calc_mse(y_left)
                mse_right = self._calc_mse(y_right)

                n_left, n_right = len(y_left), len(y_right)

                weighted_mse = (n_left/n_samples) * mse_left + (n_right/n_samples) * mse_right

                mse_reduction = parent_mse - weighted_mse

                if mse_reduction > best_mse:
                    best_mse, best_feat, best_thresh = mse_reduction, f, t

                    # Cache the actual split data to avoid re-slicing later in _build_tree
                    best_X_left, best_y_left = X[left_m], y[left_m]
                    best_X_right, best_y_right = X[right_m], y[right_m]

            # Return a dictionary containing everything needed to spawn the child nodes
        return {
            "mse_reduction": best_mse if best_feat is not None else -1,
            "feature_index": best_feat,
            "threshold": best_thresh,
            "X_left": best_X_left,
            "y_left": best_y_left,
            "X_right": best_X_right,
            "y_right": best_y_right
        }

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape

        if n_samples >= self.min_sample and depth <= self.max_depth:
            best_criteria = self._best_split(X,y)

            if best_criteria['mse_reduction'] > 0:
                left_sub = self._build_tree(best_criteria['X_left'], best_criteria['y_left'], depth + 1)
                right_sub = self._build_tree(best_criteria['X_right'], best_criteria['y_right'], depth + 1)

                return Node(
                    feature =  best_criteria['feature_index'],
                    threshold = best_criteria['threshold'],
                    left =  left_sub,
                    right =  right_sub,
                )
        leaf_value = Node(value= np.mean(y))
        return leaf_value

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def get_predictions(self, X, node):

        if node.value is not None:
            return node.value

        if X[node.feature] <= node.threshold:
            return self.get_predictions(X, node.left)
        else:
            return self.get_predictions(X, node.right)

        
    def predict(self, X):
        return np.array([self.get_predictions(x, self.root) for x in X])