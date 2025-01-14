# Linear Regression and Standard Error Calculation in Python

This repository contains Python functions to perform simple linear regression and calculate the standard error of the predictions. The implementation is lightweight and uses NumPy for numerical computations.

## Functions

### 1. `LinearRegression(x, y)`
This function calculates the coefficients \(a\) (slope) and \(b\) (intercept) of a linear regression model using the least squares method.

#### Parameters:
- `x` (list or NumPy array): Independent variable values.
- `y` (list or NumPy array): Dependent variable values.

#### Returns:
- `a` (float): The slope of the regression line.
- `b` (float): The y-intercept of the regression line.

#### Example Usage:
```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

a, b = LinearRegression(x, y)
print(f"Coefficients: a={a}, b={b}")
```

---

### 2. `StandardError(y_pred, y)`
This function calculates the standard error of the predictions based on the residuals (differences between actual and predicted values). It adjusts the calculation depending on the sample size.

#### Parameters:
- `y_pred` (list or NumPy array): Predicted values.
- `y` (list or NumPy array): Actual values.

#### Returns:
- `standard_error` (float): The standard error of the predictions.

#### Notes:
- For sample sizes \(n \geq 30\), the standard error is calculated using \( \sqrt{\frac{RSS}{n}} \), where \(RSS\) is the residual sum of squares.
- For sample sizes \(n < 30\), the formula adjusts to \( \sqrt{\frac{RSS}{n - 2}} \), accounting for degrees of freedom.
- Raises a `ValueError` if the sample size \(n \leq 2\) to prevent division by zero.

#### Example Usage:
```python
y_pred = [2.2, 3.8, 5.1, 3.9, 4.8]
y = [2, 4, 5, 4, 5]

se = StandardError(y_pred, y)
print(f"Standard Error: {se}")
```

---

## Combined Example

```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# Perform linear regression
a, b = LinearRegression(x, y)

# Generate predictions
y_pred = [a * xi + b for xi in x]

# Calculate standard error
se = StandardError(y_pred, y)

print(f"Regression Coefficients: a={a}, b={b}")
print(f"Standard Error: {se}")
```

## Requirements
- Python 3.x
- NumPy library

## Installation
To use the functions, ensure NumPy is installed:
```bash
pip install numpy
```



