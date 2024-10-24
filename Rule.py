import numpy as np

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
    grid_points = np.array([[x, y] for x in x_vals for y in y_vals])
    return grid_points

# Function to count points inside the unit square
def points_inside_square(points, tolerance=1e-9):
    inside = (np.abs(points[:, 0]) <= 0.5 + tolerance) & (np.abs(points[:, 1]) <= 0.5 + tolerance)
    return np.sum(inside)

# List of (scaling_factor, rotation_degree) pairs
transforms = [
    (1.0, 0), (1.02, 1), (1.04, 2), (1.06, 3), (1.08, 4), (1.1, 5),
    (1.11, 6), (1.14, 7), (1.16, 8), (1.17, 9), (1.18, 10),
    (1.197, 11), (1.225, 12), (1.235, 13), (1.245, 14), (1.258, 15),
    (1.27, 16), (1.28, 17), (1.29, 18), (1.3, 19), (1.31, 20),
    (1.32, 21), (1.33, 22), (1.34, 23), (1.35, 24), (1.36, 25),
    (1.37, 26), (1.38, 27), (1.39, 28), (1.40, 29), (1.405, 30),
    (1.407, 31), (1.41, 32), (1.415, 33), (1.42, 34), (1.43, 35),
    (1.432, 36), (1.434, 37), (1.436, 38), (1.438, 39), (1.44, 40),
    (1.442, 41), (1.444, 42), (1.446, 43), (1.448, 44), (1.45, 45)
]

# Define the unit square
unit_square = create_square(size=1)

# Define the blue square (initial square)
initial_blue_square = create_square(size=1)

# Function to check how many points are inside the unit square
def check_points_inside_unit_square(scale_factor, rotation_degree, perfect_square_n):
    # Generate grid points for the current perfect_square_n
    grid_points = generate_grid_points(initial_blue_square, int(np.sqrt(perfect_square_n)))

    # Rotate and scale the grid points
    rotated_grid_points = rotate_square(grid_points, angle=rotation_degree)
    scaled_rotated_grid_points = scale_square(rotated_grid_points, scale_factor=scale_factor)

    # Count the number of points inside the unit square
    inside_count = points_inside_square(scaled_rotated_grid_points)

    return inside_count

# Function to iterate through all transformations for each perfect square
def find_and_print_all_combinations():
    # Loop over n from 1 to 10
    for n in range(1, 11):
        # Calculate the perfect square size n*n
        perfect_square_n = n * n
        print(f"\nCartesian Size: {perfect_square_n}")

        # For each transformation, apply it and print the result
        for scale_factor, rotation_degree in transforms:
            # Check how many points are inside the unit square for the current transform
            inside_count = check_points_inside_unit_square(scale_factor, rotation_degree, perfect_square_n)

            # Print details for each transformation on one line
            print(f"SF: {scale_factor:.2f}, RD: {rotation_degree}, Inside Points: {inside_count}")

# Run the function to print all combinations
find_and_print_all_combinations()
