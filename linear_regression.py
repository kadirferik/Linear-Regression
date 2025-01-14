"""
Created on Tue Jan 14 22:56:46 2025

@author: kadirferik
"""
import numpy as np

def LinearRegression(x, y):
    x = np.array(x)
    y = np.array(y)
    
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_square_x = np.sum(x**2)
    sum_x_cross_y = np.sum(x * y)
    
    a = (n * sum_x_cross_y - sum_x * sum_y) / (n * sum_square_x - sum_x**2)
    b = (sum_y - a * sum_x) / n
    
    return a, b

def StandardError(y_pred, y):
    y_pred = np.array(y_pred)
    y = np.array(y)
    
    n = len(y)
    residuals = y - y_pred
    residual_sum_of_squares = np.sum(residuals ** 2)
    
    if n > 2:
        if n >= 30:
            standard_error = np.sqrt(residual_sum_of_squares / n)
        else:
            standard_error = np.sqrt(residual_sum_of_squares / (n - 2))
    else:
        raise ValueError("Sample size must be greater than 2.")
    
    return standard_error
