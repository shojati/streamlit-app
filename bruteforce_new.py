import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians

def generate_cartesian_grid(num_points_per_dimension):
    # Generate equally spaced points in a Cartesian grid
    x_values = np.linspace(0, 1, num_points_per_dimension)
    y_values = np.linspace(0, 1, num_points_per_dimension)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    points = np.vstack([x_grid.flatten(), y_grid.flatten()]).T
    return points

def rotate_and_scale_cartesian_grid(points, angle, scale, center):
    # Rotate and scale the entire grid around the center
    rotated_and_scaled_points = np.array([rotate_around_center(point, angle, scale, center) for point in points])
    return rotated_and_scaled_points

def classify_points(points, num_inside_points):
    # Classify points as inside or outside the unit square
    classifications = np.where(
        (0 <= points[:, 0]) & (points[:, 0] <= 1) & (0 <= points[:, 1]) & (points[:, 1] <= 1),
        "inside",
        "outside"
    )

    # Ensure the specified number of points are inside
    inside_indices = np.where(classifications == "inside")[0][:num_inside_points]
    classifications[:] = "outside"
    classifications[inside_indices] = "inside"

    return classifications

def visualize_grid(points, classifications, title):
    # Visualize the grid with different colors
    inside_points = points[classifications == "inside"]
    outside_points = points[classifications == "outside"]

    plt.scatter(*inside_points.T, color='red', marker='o', label='Inside')
    plt.scatter(*outside_points.T, color='blue', marker='o', label='Outside')
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

# Function to rotate a point around a center
def rotate_around_center(point, angle, scale, center):
    x_translated, y_translated = np.subtract(point, center)
    x_rot = x_translated * cos(radians(angle)) - y_translated * sin(radians(angle))
    y_rot = x_translated * sin(radians(angle)) + y_translated * cos(radians(angle))
    x_scaled_rotated, y_scaled_rotated = (x_rot * scale, y_rot * scale)
    x_final, y_final = np.add((x_scaled_rotated, y_scaled_rotated), center)
    return x_final, y_final

def rotate_and_scale_cartesian_grid(points, angle, scale, center):
    # Rotate and scale the entire grid around the center
    rotated_and_scaled_points = np.array([rotate_around_center(point, angle, scale, center) for point in points])
    return rotated_and_scaled_points

# Specify the number of points inside the unique square and the number of rotations
num_inside_points = 25
num_rotations = 36
num_points_per_dimension = 5

# Generate Cartesian grid
points = generate_cartesian_grid(num_points_per_dimension)

# Classify points
classifications = classify_points(points, num_inside_points)

# Visualize the original grid
visualize_grid(points, classifications, 'Original Square Grid (Inside and Outside)')

# Rotate and scale the Cartesian grid for different combinations of angles and scales
center = (0.5, 0.5)
scales = np.linspace(0.5, 1.5, num_rotations)  # Adjust the scaling factor range as needed

for angle in np.linspace(0, 360, num_rotations):
    for scale in scales:
        rotated_and_scaled_points = rotate_and_scale_cartesian_grid(points, angle, scale, center)

        # Classify rotated and scaled points
        classifications = classify_points(rotated_and_scaled_points, num_inside_points)

        # Visualize the rotated and scaled grid
        title = f'Rotated and Scaled Square Grid (Angle: {angle:.2f} degrees, Scale: {scale:.2f})'
        visualize_grid(rotated_and_scaled_points, classifications, title)
