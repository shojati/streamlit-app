import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians

def rotate_around_center(point, angle, scale, center):
    # Translate the point to the new center
    x_translated = point[0] - center[0]
    y_translated = point[1] - center[1]

    # Rotate and scale the translated point
    x_rot = x_translated * cos(radians(angle)) - y_translated * sin(radians(angle))
    y_rot = x_translated * sin(radians(angle)) + y_translated * cos(radians(angle))
    x_scaled_rotated = x_rot * scale
    y_scaled_rotated = y_rot * scale

    # Translate the point back to the original position
    x_final = x_scaled_rotated + center[0]
    y_final = y_scaled_rotated + center[1]

    return x_final, y_final

def classify_points(points):
    # Classify points as inside/on the border or outside the (0, 1) square
    classifications = []

    for x, y in points:
        if 0 <= x <= 1 and 0 <= y <= 1:
            classifications.append("inside")
        else:
            classifications.append("outside")

    return classifications

# Generate equally spaced points inside the square with side length sqrt(2) starting from (-sqrt(2)+1)/2
num_points = 173
x_values = np.linspace((-np.sqrt(2) + 1) / 2, (-np.sqrt(2) + 1) / 2 + np.sqrt(2), num_points)
y_values = np.linspace((-np.sqrt(2) + 1) / 2, (-np.sqrt(2) + 1) / 2 + np.sqrt(2), num_points)
x_grid, y_grid = np.meshgrid(x_values, y_values)
points = np.vstack([x_grid.flatten(), y_grid.flatten()]).T

# Classify points
classifications = classify_points(points)

# Separate points into inside and outside groups
inside_points = [point for point, classification in zip(points, classifications) if classification == "inside"]
outside_points = [point for point, classification in zip(points, classifications) if classification == "outside"]

# Visualize the original grid with different colors
plt.scatter(*zip(*inside_points), color='red', marker='o', label='Inside')
plt.scatter(*zip(*outside_points), color='blue', marker='o', label='Outside')
plt.title('Original Square Grid (Side Length: √2, Starting at (-√2+1)/2)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.xlim((-np.sqrt(2) + 1) / 2, (-np.sqrt(2) + 1) / 2 + np.sqrt(2))
plt.ylim((-np.sqrt(2) + 1) / 2, (-np.sqrt(2) + 1) / 2 + np.sqrt(2))
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

# Rotate the entire space from 0 to 360 degrees around the center (0.5, 0.5)
num_rotations = 72
center = (0.49999999999999994, 0.49999999999999994)

# Initialize a list to store results
results = []

for angle in np.linspace(0, 360, num_rotations):
    rotated_points = [rotate_around_center(point, angle, 1, center) for point in points]

    # Classify rotated points
    classifications = classify_points(rotated_points)

    # Count points in different categories
    count_inside = classifications.count("inside")
    count_outside = classifications.count("outside")

    # Calculate the percentage of inside points
    percentage_inside = count_inside / len(rotated_points) * 100

    # Append results to the list
    results.append({
        "Number of Points": len(points),
        "Angle": angle,
        "Inside Points": count_inside,
        "Outside Points": count_outside,
        "Percentage Inside": percentage_inside
    })

# Display results as a table
import pandas as pd

results_df = pd.DataFrame(results)
# Export the DataFrame to an Excel file
results_df.to_excel('result3.xlsx', index=False)

# Display the DataFrame
print(results_df)
