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
 

class DecisionTreeR:

    def __init__(self, max_depth=3, min_sample=2):
        self.max_depth = max_depth
        self.min_sample = min_sample
        self.root = None

    @staticmethod
    def _mse_from_sums(n, sum_y, sum_sq):
        """MSE from count, sum(y), and sum(y**2) — O(1) per candidate split."""
        if n == 0:
            return 0.0
        mean_y = sum_y / n
        return (sum_sq / n) - (mean_y * mean_y)

    def _calc_mse(self, y):
        if len(y) == 0:
            return 0.0
        return self._mse_from_sums(len(y), np.sum(y), np.sum(y * y))

    def _best_split(self, X, y):
        """Find best split by scanning sorted feature values (O(n log n) per feature)."""
        best_reduction = 0.0
        best_feat, best_thresh = None, None
        parent_mse = self._calc_mse(y)
        n_samples, n_features = X.shape

        total_sum = np.sum(y)
        total_sum_sq = np.sum(y * y)

        for f in range(n_features):
            order = np.argsort(X[:, f], kind="mergesort")
            x_sorted = X[order, f]
            y_sorted = y[order]

            left_sum = 0.0
            left_sum_sq = 0.0

            for i in range(1, n_samples):
                yi = y_sorted[i - 1]
                left_sum += yi
                left_sum_sq += yi * yi

                # Skip thresholds that do not change the partition
                if x_sorted[i] == x_sorted[i - 1]:
                    continue

                n_left = i
                n_right = n_samples - i
                if n_left < self.min_sample or n_right < self.min_sample:
                    continue

                right_sum = total_sum - left_sum
                right_sum_sq = total_sum_sq - left_sum_sq

                weighted_mse = (
                    (n_left / n_samples) * self._mse_from_sums(n_left, left_sum, left_sum_sq)
                    + (n_right / n_samples) * self._mse_from_sums(n_right, right_sum, right_sum_sq)
                )
                reduction = parent_mse - weighted_mse

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feat = f
                    best_thresh = (x_sorted[i - 1] + x_sorted[i]) / 2.0

        return {
            "mse_reduction": best_reduction,
            "feature_index": best_feat,
            "threshold": best_thresh,
        }

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)

        if n_samples >= self.min_sample and depth < self.max_depth:
            best = self._best_split(X, y)
            if best["feature_index"] is not None and best["mse_reduction"] > 0:
                feat, thresh = best["feature_index"], best["threshold"]
                left_m = X[:, feat] <= thresh
                right_m = ~left_m

                return Node(
                    feature=feat,
                    threshold=thresh,
                    left=self._build_tree(X[left_m], y[left_m], depth + 1),
                    right=self._build_tree(X[right_m], y[right_m], depth + 1),
                )

        return Node(value=float(np.mean(y)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.root = self._build_tree(X, y)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

