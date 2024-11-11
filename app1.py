import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

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

# Function to generate grid points inside the blue square
def generate_grid_points(square, n):
    x_vals = np.linspace(square[:, 0].min(), square[:, 0].max(), n)
    y_vals = np.linspace(square[:, 1].min(), square[:, 1].max(), n)
    return np.array([[x, y] for x in x_vals for y in y_vals])

# Function to count points inside the unit square
def points_inside_square(points, tolerance=1e-9):
    inside = (np.abs(points[:, 0]) <= 0.5 + tolerance) & (np.abs(points[:, 1]) <= 0.5 + tolerance)
    return np.sum(inside)

# Function to round n to the nearest perfect square
def round_to_perfect_square(n):
    sqrt_n = int(np.ceil(np.sqrt(n)))
    return sqrt_n * sqrt_n

# List of transformations including x and y translations
transforms = [
    # Example with translations added
    (1.55, 30, 0.1, -0.04),
    (1.58, 25, 0.15, -0.06),
    (1.55, 20, 0.15, -0.08),
    (1.5, 15, 0.15, -0.9),
    (1.35, 10, 0.1, -0.08),
    (1.1, 5, 0, 0),
    (1.14, 5, 0.03, -0.02),
    (1.19, 5, 0.05, -0.04),
    (1.25, 5, 0.08, -0.07),
    (1.32, 5, 0.12, -0.1),
    (1.32, 5, -0.1, 0.09),
    (1.25, 5, -0.07, 0.06),
    (1.19, 5, -0.04, 0.03),
    (1.14, 5, -0.02, 0.1),
    (1.18, 10, 0, 0),
    (1.22, 10, 0.03, -0.02),
    (1.27, 10, 0.05, -0.04),
    (1.33, 10, 0.08, -0.07),
    (1.39, 10, 0.12, -0.09),
    (1.36, 10, -0.09, 0.06),
    (1.32, 10, -0.07, 0.05),
    (1.27, 10, -0.04, 0.03),
    (1.22, 10, -0.02, 0.1),
    (1.25, 15, 0, 0),
    (1.28, 15, 0.03, -0.02),
    (1.32, 15, 0.05, -0.03),
    (1.37, 15, 0.08, -0.05),
    (1.43, 15, 0.12, -0.07),
    (1.41, 15, -0.09, 0.06),
    (1.38, 15, -0.07, 0.05),
    (1.33, 15, -0.04, 0.02),
    (1.29, 15, -0.02, 0.1),
    (1.31, 20, 0, 0),
    (1.33, 20, 0.03, -0.01),
    (1.36, 20, 0.05, -0.02),
    (1.42, 20, 0.08, -0.04),
    (1.46, 20, 0.12, -0.05),
    (1.45, 20, -0.09, 0.04),
    (1.42, 20, -0.07, 0.03),
    (1.38, 20, -0.04, 0.02),
    (1.34, 20, -0.02, 0.1),
    (1.345, 25, 0, 0),
    (1.39, 25, 0.03, -0.01),
    (1.41, 25, 0.05, -0.02),
    (1.46, 25, 0.08, -0.04),
    (1.52, 25, 0.12, -0.05),
    (1.49, 25, -0.09, 0.04),
    (1.47, 25, -0.07, 0.03),
    (1.42, 25, -0.04, 0.02),
    (1.39, 25, -0.02, 0.1),
    (1.4, 30, 0, 0),
    (1.42, 30, 0.03, -0.01),
    (1.44, 30, 0.05, -0.02),
    (1.49, 30, 0.08, -0.04),
    (1.55, 30, 0.12, -0.04),
    (1.54, 30, -0.09, 0.04),
    (1.51, 30, -0.07, 0.03),
    (1.47, 30, -0.04, 0.02),
    (1.44, 30, -0.02, 0.1),
    (1.42, 35, 0, 0),
    (1.45, 35, 0.03, -0.01),
    (1.48, 35, 0.05, -0.02),
    (1.53, 35, 0.08, -0.02),
    (1.58, 35, 0.12, -0.04),
    (1.54, 35, -0.09, 0.02),
    (1.52, 35, -0.07, 0.02),
    (1.48, 35, -0.04, 0.01),
    (1.44, 35, -0.02, 0.01),
    (1.43, 40, 0, 0),
    (1.46, 40, 0.03, -0.01),
    (1.5, 40, 0.05, -0.02),
    (1.54, 40, 0.08, -0.02),
    (1.58, 40, 0.12, -0.02),
    (1.56, 40, -0.09, 0.01),
    (1.53, 40, -0.07, 0),
    (1.49, 40, -0.04, 0.01),
    (1.47, 40, -0.02, 0),
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

# Define the unit square
unit_square = create_square(size=1)

# Define the blue square (initial square)
initial_blue_square = create_square(size=1)

# Function to check how many points are inside the unit square
def check_if_n_points_inside_unit_square(scale_factor, rotation_degree, x_translation, y_translation, perfect_square_n):
    # Generate grid points for the current perfect_square_n
    grid_points = generate_grid_points(initial_blue_square, int(np.sqrt(perfect_square_n)))

    # Rotate, scale, and then translate the grid points
    rotated_grid_points = rotate_square(grid_points, angle=rotation_degree)
    scaled_rotated_grid_points = scale_square(rotated_grid_points, scale_factor=scale_factor)
    translated_points = scaled_rotated_grid_points + np.array([x_translation, y_translation])

    # Count the number of points inside the unit square
    inside_count = points_inside_square(translated_points)

    return inside_count, translated_points

# Function to iterate through all transformations for each perfect square
def find_and_plot_all_combinations(n):
    perfect_square_n = round_to_perfect_square(n)
    max_search_limit = 10 * n

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(unit_square[:, 0], unit_square[:, 1], 'r-', label='Unit Square', linewidth=2)

    while perfect_square_n <= max_search_limit:
        print(f"Cartesian Size: {perfect_square_n}")

        for scale_factor, rotation_degree, x_translation, y_translation in transforms:
            inside_count, transformed_grid_points = check_if_n_points_inside_unit_square(
                scale_factor, rotation_degree, x_translation, y_translation, perfect_square_n
            )

            print(f"SF: {scale_factor:.2f}, RD: {rotation_degree}, X: {x_translation}, Y: {y_translation}, Inside Points: {inside_count}")

            if inside_count == n:
                print(f"\n** FOUND DESIRED N = {n} for SF: {scale_factor:.2f}, RD: {rotation_degree}, X: {x_translation}, Y: {y_translation} **")
                return scale_factor, rotation_degree, x_translation, y_translation, fig, ax, transformed_grid_points

        sqrt_perfect_square_n = int(np.sqrt(perfect_square_n)) + 1
        perfect_square_n = sqrt_perfect_square_n * sqrt_perfect_square_n
        print(f"Increasing Cartesian Size to: {perfect_square_n}")

    return None, None, None, None, None, None, None

# Streamlit UI elements for input
st.title("Distribution of N Points inside Unit Square")

col1, col2 = st.columns([1, 2])

with col1:
    n = st.number_input("Enter N", min_value=1, step=1)

if n:
    scale_factor, rotation_degree, x_translation, y_translation, fig, ax, transformed_grid_points = find_and_plot_all_combinations(n)

    if scale_factor is not None:
        with col1:
            st.markdown(
                f"<h2 style='text-align: left; color: green;font-size: 20px;'>N = {n} <br> SF: {scale_factor:.2f} <br> RD: {rotation_degree}<br> X: {x_translation} <br> Y: {y_translation}</h2>",
                unsafe_allow_html=True)
    else:
        with col1:
            st.markdown(f"<h2 style='text-align: left; color: red;font-size: 20px;'> No solution found for N = {n}. </h2>", unsafe_allow_html=True)

    with col2:
        if fig and ax:
            ax.scatter(transformed_grid_points[:, 0], transformed_grid_points[:, 1], color='g', s=10, alpha=0.5)
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.axhline(0, color='black', linewidth=0.5)
            plt.axvline(0, color='black', linewidth=0.5)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title('Distribution of Points in Unit Square')
            st.pyplot(fig)
