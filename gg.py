import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    (1.1, 5, 0, 0),
    (1.14, 5, 0.03, -0.02),
    (1.19, 5, 0.05, -0.04),
    (1.25, 5, 0.08, -0.07),
    (1.32, 5, 0.12, -0.11),
    (1.32, 5, -0.01, 0.11),
    (1.25, 5, -0.07, 0.06),
    (1.19, 5, -0.04, 0.04),
    (1.14, 5, -0.02, 0.02),
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
    (1.31, 20, 0, 0),
    (1.33, 20, 0.03, -0.01),
    (1.36, 20, 0.05, -0.02),
    (1.42, 20, 0.08, -0.04),
    (1.46, 20, 0.12, -0.05),
    (1.45, 20, -0.09, 0.04),
    (1.42, 20, -0.07, 0.03),
    (1.38, 20, -0.04, 0.02),
    (1.34, 20, -0.02, 0.01),
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
    (1.4, 30, 0, 0),
    (1.38, 30, 0, 0),
    (1.42, 30, 0.03, -0.01),
    (1.44, 30, 0.05, -0.02),
    (1.49, 30, 0.08, -0.04),
    (1.55, 30, 0.12, -0.04),
    (1.54, 30, -0.09, 0.04),
    (1.51, 30, -0.07, 0.03),
    (1.47, 30, -0.04, 0.02),
    (1.44, 30, -0.02, 0.01),
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

    # You can add more transformations as needed
]

# Define the unit square
unit_square = create_square(size=1)

# Define the blue square (initial square)
initial_blue_square = create_square(size=1)

# Function to check how many points are inside the unit square
def check_if_n_points_inside_unit_square(scale_factor, rotation_degree, x_translation, y_translation, perfect_square_n):
    grid_points = generate_grid_points(initial_blue_square, int(np.sqrt(perfect_square_n)))
    rotated_grid_points = rotate_square(grid_points, angle=rotation_degree)
    scaled_rotated_grid_points = scale_square(rotated_grid_points, scale_factor=scale_factor)
    translated_points = scaled_rotated_grid_points + np.array([x_translation, y_translation])
    inside_count = points_inside_square(translated_points)
    return inside_count, translated_points

# Animation function
def animate_combinations(n):
    perfect_square_n = round_to_perfect_square(n)
    max_search_limit = 10 * n

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(unit_square[:, 0], unit_square[:, 1], 'r-', label='Unit Square', linewidth=2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.legend()

    scatter = ax.scatter([], [], color='blue', s=10, alpha=0.5)

    transformation_data = []

    # Iterate through transformations
    while perfect_square_n <= max_search_limit:
        for scale_factor, rotation_degree, x_translation, y_translation in transforms:
            inside_count, transformed_points = check_if_n_points_inside_unit_square(
                scale_factor, rotation_degree, x_translation, y_translation, perfect_square_n
            )
            transformation_data.append((scale_factor, rotation_degree, x_translation, y_translation, transformed_points))
            if inside_count == n:
                print(f"Found: SF={scale_factor}, RD={rotation_degree}, X={x_translation}, Y={y_translation}")
                break
        else:
            sqrt_perfect_square_n = int(np.sqrt(perfect_square_n)) + 1
            perfect_square_n = sqrt_perfect_square_n * sqrt_perfect_square_n
            continue
        break

    def update(frame):
        scale_factor, rotation_degree, x_translation, y_translation, transformed_points = transformation_data[frame]
        scatter.set_offsets(transformed_points)
        ax.set_title(
            f"SF: {scale_factor:.2f}, RD: {rotation_degree}, X: {x_translation}, Y: {y_translation}",
            fontsize=12
        )
        return scatter,

    ani = FuncAnimation(fig, update, frames=len(transformation_data), blit=False, repeat=False)
    plt.show()

# Main execution
if __name__ == "__main__":
    n = int(input("Enter the value of N: "))
    animate_combinations(n)
