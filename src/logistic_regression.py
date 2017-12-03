import numpy as np

def predict_proba(X, coefs):
    '''
    Inputs:
    X: ndarray, shape (sample_cnt, features_cnt)
    coefs: ndarray, shape (features_cnt, )
    
    Returns:
    predicted_probabilities, shape (sample_cnt, )
    '''
    return 1/(1 + np.exp(-np.dot(X, coefs)))

def predict(X, coefs, threshold=0.5):
    '''
    Inputs:
    X: ndarray, shape (sample_cnt, features_cnt)
    coefs: ndarray, shape (features_cnt, )
    threshold: float, default 0.5
    
    Returns:
    prediction_class, boolean
    '''
    proba = predict_proba(X, coefs)
    return proba >= threshold

def cost_batch(X, y, coefs, lam=0.0, has_intercept=True):
    '''
    Inputs:
    X: ndarray, shape (sample_cnt, features_cnt)
    y: ndarray, shape (sample_cnt, 1)
    coefs: ndarray, shape (features_cnt, )
    lam: float, default 0.0
        Regularization term
    has_intercept, default True
        whether the first element of coefs is an intercept parameter
    
    Returns:
    log_cost_function_at_coefs, float
    '''
    p = predict_proba(X, coefs)
    ridge_penalty = np.sum(coefs**2)
    if has_intercept:
        ridge_penalty -= coefs[0]**2
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + lam * ridge_penalty

def cost_SGD(x, y, coefs):
    '''
    Inputs:
    x: ndarray, shape (features_cnt, )
    y: float
    coefs: ndarray, shape (features_cnt, )
    
    Returns:
    one_point_log_cost_function_at_coefs, float
    '''
    
    log_odds = np.sum(x * coefs)
    #import pdb;pdb.set_trace()
    p = 1 / (1 + np.exp(-log_odds))
    return (- y * np.log(p) - (1 - y) * np.log(1 - p))

def gradient_batch(X, y, coefs, lam=0.0, has_intercept=True):
    '''
    Input:
    X: ndarray, shape (sample_cnt, features_cnt)
    y: ndarray, shape (sample_cnt, 1)
    coefs: ndarray, shape (features_cnt, )
    lam: float, default 0.0
        Regularization term
    has_intercept, default True
        whether the first element of coefs is an intercept parameter
    
    Returns:
    whole_batch_cost_function_gradient, shape (features_cnt, )
    '''
    prob = predict_proba(X, coefs)
    ridge_grad = 2 * coefs
    if has_intercept:
        ridge_grad[0] = 0.0
    return np.dot(X.T, (prob - y)) + lam * ridge_grad
    
def gradient_SGD(x, y, coefs):
    '''
    Inputs:
    x: ndarray, shape (features_cnt, )
    y: float
    coefs: ndarray, shape (features_cnt, )
    
    Returns:
    one_point_cost_function_gradient, shape (features_cnt, )
    '''
    log_odds = np.sum(x * coefs)
    p = 1 / 1 + np.exp(-log_odds)
    return (p - y) * x

