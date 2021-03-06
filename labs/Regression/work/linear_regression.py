import numpy as np
import pandas as pd

############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here
    # TODO  : add tests
    #####################################################
    
    N = len(y)
    
    err = sum([(np.dot(w, X[n]) - y[n])**2  for n in range(N)])/N
   
    return err

###### Part 1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing features
  - y: A numpy array of shape (num_samples, ) containing labels
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here       
  # TODO  : Add tests
  #####################################################

  w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
  
  return w


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here
  # TODO  : Add tests
  #####################################################

    x2 = np.dot(np.transpose(X), X)
    w = np.dot(np.dot(np.linalg.inv(np.add(x2, lambd * np.identity(len(x2)))), np.transpose(X)), y)

    return w

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    #####################################################
    # TODO 5: Fill in your code here
    # TODO  : Add tests
    #####################################################

    lambdas = [2**(-i) for i in range(15)]

    bestlambda = None
    miniMeanSqErr = None

    for lambdaVal in lambdas:
        cur_miniMeanSqErr = mean_square_error(regularized_linear_regression(Xtrain, ytrain, lambdaVal), Xval, yval)

        if miniMeanSqErr is None or cur_miniMeanSqErr < miniMeanSqErr:
            miniMeanSqErr = cur_miniMeanSqErr
            bestlambda = lambdaVal

    return bestlambda
    

###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    # TODO 6: Fill in your code here                    #
    #####################################################		

    cols = len(X[0])

    #white = np.insert(mapped_X, mapped_X.shape[1], white[:, -1], axis=1)

    for i in range(2, p + 1):
        for col in range(cols):
            column = X[:, col]
            column = np.power(column, i)
            X = np.insert(X, X.shape[1], column, axis=1)

    return X

"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

