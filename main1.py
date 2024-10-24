import numpy as np

import math

def calculate_diagonal_intersection(bottom_left_vertex, side_length):
    # Assuming bottom_left_vertex is a tuple containing coordinates of the bottom-left vertex
    # and side_length is the length of each side of the square

    # Calculate other vertices based on the bottom-left vertex and side length
    x1, y1 = bottom_left_vertex
    x2, y2 = x1 + side_length, y1
    x3, y3 = x1 + side_length, y1 + side_length
    x4, y4 = x1, y1 + side_length

    # Calculate intersection point
    intersection_x = (x1 + x3) / 2
    intersection_y = (y1 + y3) / 2

    return (x1, y1), (x2, y2), (x3, y3), (x4, y4), (intersection_x, intersection_y)

# Example usage
bottom_left_vertex = ((-np.sqrt(2) + 1)/2, (-np.sqrt(2) + 1)/2)
side_length = math.sqrt(2)
vertices_and_intersection = calculate_diagonal_intersection(bottom_left_vertex, side_length)

print("Vertices and Intersection Point:")
for vertex in vertices_and_intersection:
    print(vertex)






