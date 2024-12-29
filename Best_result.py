import numpy as np
import pandas as pd

# Function to create the square's vertices
def create_square(size=1):
    half_size = size / 2
    return np.array([
        [-half_size, -half_size],
        [-half_size, half_size],
        [half_size, half_size],
        [half_size, -half_size],
        [-half_size, -half_size]  # Closing the square
    ])

# Function to rotate the square by a certain angle (in degrees)
def rotate_square(square, angle):
    theta = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return square @ rotation_matrix.T

# Function to scale the square by a scaling factor
def scale_square(square, scale_factor):
    return square * scale_factor

# Function to generate grid points inside the square
def generate_grid_points(square, n):
    x_vals = np.linspace(square[:, 0].min(), square[:, 0].max(), n)
    y_vals = np.linspace(square[:, 1].min(), square[:, 1].max(), n)
    return np.array([[x, y] for x in x_vals for y in y_vals])

# Function to count points inside the unit square
def points_inside_square(points, tolerance=1e-9):
    inside = (np.abs(points[:, 0]) <= 0.5 + tolerance) & (np.abs(points[:, 1]) <= 0.5 + tolerance)
    return np.sum(inside)

# List of transformations
transforms = [(1.0, 0, 0, 0),
    (1.05, 2, 0, 0),
    (1.1, 2, 0.03, -0.03),
    (1.14, 2, 0.05, -0.05),
    (1.2, 2, 0.08, -0.08),
    (1.28, 2, 0.12, -0.12),
    (1.25, 2, -0.1, 0.1),
    (1.19, 2, -0.07, 0.07),
    (1.13, 2, -0.04, 0.04),
    (1.1, 2, -0.02, 0.025),
    (1.1, 5, 0, 0),
    (1.14, 5, 0.03, -0.02),
    (1.19, 5, 0.05, -0.04),
    (1.25, 5, 0.08, -0.07),
    (1.32, 5, 0.12, -0.11),
    (1.32, 5, -0.01, 0.11),
    (1.25, 5, -0.07, 0.06),
    (1.19, 5, -0.04, 0.04),
    (1.14, 5, -0.02, 0.02),
    (1.13, 7, 0, 0),
    (1.18, 7, 0.03, -0.03),
    (1.21, 7, 0.05, -0.04),
    (1.27, 7, 0.08, -0.07),
    (1.33, 7, 0.12, -0.09),
    (1.3, 7, -0.09, 0.07),
    (1.26, 7, -0.07, 0.055),
    (1.21, 7, -0.04, 0.03),
    (1.17, 7, -0.02, 0.02),
    (1.18, 10, 0, 0),
    (1.17, 10, 0, 0),
    (1.22, 10, 0.03, -0.02),
    (1.27, 10, 0.05, -0.04),
    (1.33, 10, 0.08, -0.07),
    (1.39, 10, 0.12, -0.09),
    (1.35, 10, 0.1, -0.08),
    (1.36, 10, -0.09, 0.06),
    (1.32, 10, -0.07, 0.05),
    (1.27, 10, -0.04, 0.03),
    (1.22, 10, -0.02, 0.02),
    (1.21, 13, 0, 0),
    (1.26, 13, 0.03, -0.02),
    (1.29, 13, 0.05, -0.03),
    (1.35, 13, 0.08, -0.055),
    (1.41, 13, 0.12, -0.075),
    (1.37, 13, -0.09, 0.055),
    (1.34, 13, -0.07, 0.045),
    (1.28, 13, -0.04, 0.025),
    (1.25, 13, -0.02, 0.01),
    (1.25, 15, 0, 0),
    (1.24, 15, 0, 0),
    (1.28, 15, 0.03, -0.02),
    (1.32, 15, 0.05, -0.03),
    (1.37, 15, 0.08, -0.05),
    (1.43, 15, 0.12, -0.07),
    (1.41, 15, -0.09, 0.06),
    (1.38, 15, -0.07, 0.05),
    (1.33, 15, -0.04, 0.02),
    (1.29, 15, -0.02, 0.01),
    (1.27, 15, -0.02, 0.01),
    (1.27, 17, 0, 0),
    (1.3, 17, 0.03, -0.014),
    (1.33, 17, 0.05, -0.025),
    (1.38, 17, 0.08, -0.045),
    (1.45, 17, 0.12, -0.066),
    (1.41, 17, -0.09, 0.05),
    (1.38, 17, -0.07, 0.04),
    (1.33, 17, -0.04, 0.02),
    (1.3, 17, -0.02, 0.01),
    (1.31, 20, 0, 0),
    (1.33, 20, 0.03, -0.01),
    (1.36, 20, 0.05, -0.02),
    (1.42, 20, 0.08, -0.04),
    (1.47, 20, 0.12, -0.05),
    (1.45, 20, -0.09, 0.04),
    (1.42, 20, -0.07, 0.03),
    (1.38, 20, -0.04, 0.02),
    (1.34, 20, -0.02, 0.01),
    (1.315, 22, 0, 0),
    (1.36, 22, 0.03, -0.015),
    (1.39, 22, 0.05, -0.022),
    (1.44, 22, 0.08, -0.04),
    (1.5, 22, 0.12, -0.05),
    (1.46, 22, -0.09, 0.04),
    (1.43, 22, -0.07, 0.03),
    (1.39, 22, -0.04, 0.02),
    (1.35, 22, -0.02, 0.01),
    (1.345, 25, 0, 0),
    (1.39, 25, 0.03, -0.01),
    (1.41, 25, 0.05, -0.02),
    (1.46, 25, 0.08, -0.04),
    (1.52, 25, 0.12, -0.05),
    (1.58, 25, 0.15, -0.06),
    (1.49, 25, -0.09, 0.04),
    (1.47, 25, -0.07, 0.03),
    (1.42, 25, -0.04, 0.02),
    (1.39, 25, -0.02, 0.01),
    (1.38, 25, -0.02, 0.01),
    (1.36, 27, 0, 0),
    (1.4, 27, 0.03, -0.01),
    (1.44, 27, 0.05, -0.02),
    (1.48, 27, 0.08, -0.03),
    (1.53, 27, 0.12, -0.04),
    (1.5, 27, -0.09, 0.03),
    (1.46, 27, -0.07, 0.02),
    (1.42, 27, -0.04, 0.01),
    (1.39, 27, -0.02, 0.005),
    (1.4, 30, 0, 0),
    (1.38, 30, 0, 0),
    (1.42, 30, 0.03, -0.01),
    (1.45, 30, 0.05, -0.02),
    (1.49, 30, 0.08, -0.03),
    (1.55, 30, 0.12, -0.04),
    (1.54, 30, -0.09, 0.04),
    (1.51, 30, -0.07, 0.03),
    (1.47, 30, -0.04, 0.02),
    (1.44, 30, -0.02, 0.01),
    (1.39, 32, 0, 0),
    (1.43, 32, 0.03, -0.01),
    (1.46, 32, 0.05, -0.02),
    (1.5, 32, 0.08, -0.02),
    (1.56, 32, 0.12, -0.03),
    (1.53, 32, -0.09, 0.02),
    (1.49, 32, -0.07, 0.015),
    (1.45, 32, -0.04, 0.01),
    (1.42, 32, -0.02, 0.005),
    (1.42, 35, 0, 0),
    (1.45, 35, 0.03, -0.01),
    (1.48, 35, 0.05, -0.02),
    (1.53, 35, 0.08, -0.02),
    (1.58, 35, 0.12, -0.04),
    (1.54, 35, -0.09, 0.02),
    (1.52, 35, -0.07, 0.02),
    (1.48, 35, -0.04, 0.01),
    (1.44, 35, -0.02, 0.01),
    (1.41, 37, 0, 0),
    (1.45, 37, 0.03, -0.01),
    (1.48, 37, 0.05, -0.01),
    (1.52, 37, 0.08, -0.01),
    (1.58, 37, 0.12, -0.02),
    (1.55, 37, -0.09, 0.01),
    (1.52, 37, -0.07, 0.01),
    (1.47, 37, -0.04, 0.005),
    (1.44, 37, -0.02, 0),
    (1.43, 40, 0, 0),
    (1.46, 40, 0.03, -0.01),
    (1.5, 40, 0.05, -0.02),
    (1.54, 40, 0.08, -0.02),
    (1.59, 40, 0.12, -0.02),
    (1.56, 40, -0.09, 0.01),
    (1.53, 40, -0.07, 0),
    (1.49, 40, -0.04, 0.01),
    (1.47, 40, -0.02, 0),
    (1.425, 42, 0, 0),
    (1.46, 42, 0.03, -0.005),
    (1.49, 42, 0.05, -0.005),
    (1.54, 42, 0.08, -0.01),
    (1.58, 42, 0.12, -0.01),
    (1.55, 42, -0.09, 0),
    (1.52, 42, -0.07, 0),
    (1.49, 42, -0.04, 0),
    (1.46, 42, -0.02, 0),
    (1.43, 45, 0, 0),
    (1.47, 45, 0.03, -0.01),
    (1.5, 45, 0.05, -0.01),
    (1.55, 45, 0.08, -0.01),
    (1.59, 45, 0.12, -0.01),
    (1.57, 45, -0.09, 0),
    (1.54, 45, -0.07, 0),
    (1.5, 45, -0.04, 0),
    (1.47, 45, -0.02, 0.1)
]

# Function to check points inside unit square after transformation
def check_if_n_points_inside_unit_square(scale_factor, rotation_degree, x_translation, y_translation, n_points):
    # Define the initial square and grid points
    initial_square = create_square(size=1)
    grid_points = generate_grid_points(initial_square, int(np.sqrt(n_points)))

    # Apply transformations
    rotated_grid_points = rotate_square(grid_points, angle=rotation_degree)
    scaled_rotated_grid_points = scale_square(rotated_grid_points, scale_factor=scale_factor)
    transformed_points = scaled_rotated_grid_points + np.array([x_translation, y_translation])

    # Count points inside the unit square
    inside_count = points_inside_square(transformed_points)

    return inside_count

# Main function to find and save results to Excel
def find_and_save_combinations_to_excel(start_n, end_n, filename="transformations_new_results5.xlsx"):
    results = []

    for n in range(start_n, end_n + 1):
        n_points = n * n  # Total number of points in the rotated square

        for scale_factor, rotation_degree, x_translation, y_translation in transforms:
            inside_count = check_if_n_points_inside_unit_square(
                scale_factor, rotation_degree, x_translation, y_translation, n_points
            )
            results.append({
                "n^2 (Total Points)": n_points,
                "Inside Points": inside_count,
                "Scaling Factor": scale_factor,
                "Rotation Degree": rotation_degree,
                "Translation X": x_translation,
                "Translation Y": y_translation
            })

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Save to Excel
    df.to_excel(filename, index=False)
    print(f"Results saved to {filename}")

# Run the function for N from 1 to 10
find_and_save_combinations_to_excel(401, 500)
