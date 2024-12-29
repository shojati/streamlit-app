import numpy as np
from scipy.optimize import curve_fit



# Example data
rotation_degrees = np.array([30, 25, 20, 15, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5])
tx_values = np.array([0.1, 0.15, 0.15, 0.15, 0.1, 0, 0.03, 0.05, 0.08, 0.12, -0.1, -0.07, -0.04, -0.02])
ty_values = np.array([-0.04, -0.06, -0.08, -0.09, -0.08, 0, -0.02, -0.04, -0.07, -0.1, 0.09, 0.06, 0.03, 0.1])
scaling_factors = np.array([1.55, 1.58, 1.55, 1.5, 1.35, 1.1, 1.14, 1.19, 1.25, 1.32, 1.32, 1.25, 1.19, 1.14])


# Define the model function
def scaling_model(vars, a, b, c, d, e, f):
    rotation, tx, ty = vars
    return a * np.cos(b * np.radians(rotation)) + c * rotation + d * tx + e * ty + f

# Prepare the independent variable array
independent_vars = (rotation_degrees, tx_values, ty_values)

# Initial guess for parameters [a, b, c, d, e, f]
initial_guess = [1, 0.1, 0.1, 0.1, 0.1, 0.1]

# Perform curve fitting
params, covariance = curve_fit(scaling_model, independent_vars, scaling_factors, p0=initial_guess)

# Extract fitted parameters
a, b, c, d, e, f = params
print(f"Fitted parameters: a={a}, b={b}, c={c}, d={d}, e={e}, f={f}")

# Calculate fitted scaling values and residuals
fitted_scaling = scaling_model((rotation_degrees, tx_values, ty_values), *params)
residuals = scaling_factors - fitted_scaling
print("Residuals:", residuals)

# Optionally, calculate RMSE or R^2 to assess the fit
rmse = np.sqrt(np.mean(residuals**2))
print("Root Mean Square Error (RMSE):", rmse)
