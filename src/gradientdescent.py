import numpy as np

class GradientDescent(object):
    
    def __init__(self, cost, gradient, predict_func, 
                 alpha=0.01,
                 step_size=None,
                 fit_intercept = False,
                 num_iterations=10000):
        '''
        Inputs
        cost: The cost function to be minimized.
        gradient: The gradient of the cost function.
        predict_func: A function to make predictions after the optimization has
            converged.
        alpha: The learning rate.
        fit_intercept: default False
        whether to fit intercept parameter
        step_size: float
        stop criteria, minimum cost decrease
        num_iterations: Number of iterations to use in the descent.
        '''
        # Initialize coefficients in run method once you know how many features
        # you have.
        self.coeffs = None
        self.cost = cost
        self.gradient = gradient
        self.predict_func = predict_func
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.step_size = step_size
    
    def fit(self, X, y):
        '''
        Input:
        X: ndarray, shape (sample_cnt, features_cnt)
        y: ndarray, shape (sample_cnt, 1)
        
        Returns:
        self
        '''
        if self.fit_intercept:
            X = self.add_intercept(X)
        self.coeffs = np.zeros(X.shape[1])
        costs = []
        for i in range(self.num_iterations):
            self.coeffs = self.coeffs - self.alpha * self.gradient(X, y, self.coeffs, has_intercept=self.fit_intercept)
            cost = self.cost(X, y, self.coeffs, has_intercept=self.fit_intercept)
            costs.append(cost)
            if i >= 2 and (self.step_size and np.abs(cost - costs[-2]) <= self.step_size):
                break
        #import pdb;pdb.set_trace()
        return self
    
    def fit_SGD(self, X, y):
        '''
        Input:
        X: ndarray, shape (sample_cnt, features_cnt)
        y: ndarray, shape (sample_cnt, 1)
        
        Returns:
        self
        '''
        self.coeffs = np.zeros(X.shape[1])
        idxs = np.random.permutation(X.shape[0])
        X = X[idxs, :]
        y = y[idxs]
        costs = []
        converged = False
        for i in xrange(self.num_iterations):
            for j in xrange(X.shape[0]):
                grad = self.gradient(X[j, :], y[j], self.coeffs)
                self.coeffs = self.coeffs - self.alpha * grad
                cost = self.cost(X[j, :], y[j], self.coeffs)
                costs.append(cost)
            if self.step_size:
                prior_sum = np.sum(costs[(-2 * X.shape[0]):(-X.shape[0])])
                current_sum = np.sum(costs[(-X.shape[0]):])
                converged = np.abs(prior_sum - current_sum) <= self.step_size
            if (i >= 2 and converged):
                break
        return self
    
    def predict(self, X):
        """Call self.predict_func to return predictions.
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            Predictor data to make predictions for.
        Returns
        -------
        preds: ndarray, shape (n_samples)
            Array of predictions.
        """
        if self.fit_intercept:
            X = self.add_intercept(X)
        return self.predict_func(X, self.coeffs)
    
    def add_intercept(self, X):
        """Add an intercept column to a matrix X.
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
        Returns
        -------
        X: ndarray, shape (n_samples, n_features + 1)
            The original matrix X, but with a constant columns of 1's appended.
        """
        ones = np.ones(X.shape[0]).reshape(-1, 1)
        return np.concatenate([ones, X], axis=1)