import numpy as np
import pandas as pd

class LinearR:
    def __init__(self, X, y, iterations=100, tolerance=1e-6) -> None:
        self.train_var = X
        self.pred_var = y
        self.theta = None
        self.lr = 0.001
        self.iters = iterations
        self.tol = tolerance
        self.total_cost = []

    def matrix_conversion(self, X, y, training=True):
        bias = pd.Series(np.ones(X.shape[0]))
        dep = X.to_numpy()
        X = np.column_stack((bias, dep))

        if not training:
            return X
        else:
            if y is None or (hasattr(y, "empty") and y.empty):
                raise ValueError(f"{y} is empty.")
                
            y = y.to_numpy() 

            # return [fin_dep.shape, self.train_var.shape]
            return X, y


    def bkp_matrix_conversion(self, X, y=None, training=True):
        bias = pd.Series(np.ones(X.shape[0]))
        dep = X.to_numpy()
        X = np.column_stack((bias, dep))            
        y = y.to_numpy() 

        # return [fin_dep.shape, self.train_var.shape]
        return X, y

    def model_parameters(self, data):
        self.theta = np.random.uniform(-1, 1, size=data.shape[1])
   

    def forward_pass(self, X, theta):
        return np.dot(X, theta)

    def cost(self, X, y, theta):
        preds = self.forward_pass(X, theta)
        error =  preds - y
        return error
    
    def mse(self, y, error):
        j0 = (1 / (2 * len(y))) * np.sum(np.square(error))
        return j0

    def grad_des(self, X, error):
        gr = 1/len(X) * np.dot(X.T, error) 
        return gr

    def fit(self, X, y):
        print(type(X), type(y))

        X, y = self.matrix_conversion(X, y)
        print(X.shape, y.shape)
        self.model_parameters(X)

        # Assert that X and y are numpy arrays
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        # Check that X and y have compatible sizes
        assert X.shape[0] == y.shape[0], f"X and y must have the same number of samples, got {X.shape[0]} and {y.shape[0]}"
        print("Training Started")
        for i in range(self.iters):
            # preds = self.forward_pass(X, theat)
            err = self.cost(X, y, self.theta)

            grad = self.grad_des(X, err)
            self.theta=  self.theta - self.lr * grad

            residual_err = self.cost(X, y, self.theta)
            new_cos = self.mse(y, residual_err)
            # print("Residual Error -- >", residual_err)
            # print("New Error -- >", new_cos)

            if i > 0 and abs(self.total_cost[-1] - new_cos) < self.tol:
                break
                
            self.total_cost.append(new_cos)
            if i % 10000 == 0:
                print(f" \nIteration #{i} with cost {new_cos}")
                # print(f" Iteration #{i} Cost difference to Tolerence: {self.total_cost[-1] - new_cos}   <---->   {self.tol}")

        print("Training Ended!")

    def predict(self, X):
        X = self.matrix_conversion(X, 0, training=False)
        # Assert that X is a numpy array
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        # Check that X has the correct number of features
        assert X.shape[1] == self.theta.shape[0], f"Number of features in X ({X.shape[1]}) must match the number of model parameters ({self.theta.shape[0]})"
        return self.forward_pass(X, self.theta)

if __name__=="__main__":
    pass