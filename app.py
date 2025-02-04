import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# Custom CSS for background color and space between elements
st.markdown(
    """
    <style>
    body {
        background-color: #d3d3d3 !important;  /* Force background color to light gray */
    }
    .stButton>button {
        margin-top: 20px;  /* Adds space above buttons */
        margin-bottom: 20px;  /* Adds space below buttons */
    }
    .stSlider>div>div>div>input {
        margin-top: 20px;  /* Adds space above slider */
    }
    </style>
    """, unsafe_allow_html=True
)


# Layout: Split the page into 2 rows
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
    (1.47, 45, -0.02, 0.1),
    (2.27, 7, 0.08, -0.07)
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
    step_count = 0  # Counter for the number of transformations

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(unit_square[:, 0], unit_square[:, 1], 'r-', label='Unit Square', linewidth=2)

    while perfect_square_n <= max_search_limit:
        print(f"Cartesian Size: {perfect_square_n}")

        for scale_factor, rotation_degree, x_translation, y_translation in transforms:
            # Increment step count
            step_count += 1

            # Process the transformation
            inside_count, transformed_grid_points = check_if_n_points_inside_unit_square(
                scale_factor, rotation_degree, x_translation, y_translation, perfect_square_n
            )

            print(f"Step {step_count}: SF: {scale_factor:.2f}, RD: {rotation_degree}, "
                  f"X: {x_translation}, Y: {y_translation}, Inside Points: {inside_count}, P_N: {perfect_square_n}")

            if inside_count == n:
                print(f"\n** FOUND DESIRED N = {n} in {step_count} steps. "
                      f"SF: {scale_factor:.2f}, RD: {rotation_degree}, "
                      f"X: {x_translation}, Y: {y_translation} **")
                return scale_factor, rotation_degree, x_translation, y_translation, fig, ax, transformed_grid_points, perfect_square_n, step_count

            # Increase grid size to the next perfect square
        sqrt_perfect_square_n = int(np.sqrt(perfect_square_n)) + 1
        perfect_square_n = sqrt_perfect_square_n * sqrt_perfect_square_n
        print(f"Increasing Cartesian Size to: {perfect_square_n}")

    return None, None, None, None, None, None, None, None, step_count


# Streamlit UI elements for input
st.title("Distribution of N Points inside Unit Square")

col1, col2 = st.columns([1, 1])

# Streamlit UI elements for input and output in the sidebar

# Sidebar for input parameters
st.sidebar.header("Input Parameters")
n = st.sidebar.number_input("Enter the value of N:", min_value=1, max_value=5000, value=1, step=1)

# Initialize placeholders for results
result_text = ""
transformation_info = ""

if n:
    (scale_factor, rotation_degree, x_translation, y_translation, fig, ax, transformed_grid_points, perfect_square_n,
     step_count) = find_and_plot_all_combinations(n)

    if scale_factor is not None:
        # Format the transformation result
        # Format the transformation result with HTML for better styling
        transformation_info = f"""
            <h2 style='text-align: left; color: green; font-size: 20px;'>
                Transformation Result for N = {n}
            </h2>
            <ul style='list-style-type: none; padding: 0;'>
                <li style='font-size: 16px;'><strong>Scaling Factor :</strong> {scale_factor:.2f}</li>
                <li style='font-size: 16px;'><strong>Rotation Degree :</strong> {rotation_degree}°</li>
                <li style='font-size: 16px;'><strong>Translation_X:</strong> {x_translation}</li>
                <li style='font-size: 16px;'><strong>Translation_Y:</strong> {y_translation}</li>
                <li style='font-size: 16px;'><strong>Cartesian grid Size:</strong> {perfect_square_n}</li>
            </ul>
        """

    else:
        result_text = f"No solution found for N = {n}."

    # Show results in the sidebar
    with st.sidebar:
        if transformation_info:
            st.markdown(f"<h2 style='text-align: left; color: green;font-size: 20px;'>{transformation_info}</h2>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='text-align: left; color: red;font-size: 20px;'>{result_text}</h2>",
                        unsafe_allow_html=True)

    # Main plot display
    with col1:
        if fig and ax:
            # Plot the transformed grid points
            ax.scatter(transformed_grid_points[:, 0], transformed_grid_points[:, 1], color='b', s=10, alpha=0.5)

            # Show only the unit square, which lies between -0.5 and 0.5 on both axes
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)

            # Optionally, remove the axis lines outside the square (set equal aspect ratio)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)

            # Make sure the plot remains square (equal aspect ratio)
            ax.set_aspect('equal', adjustable='box')

            # Optionally add title for context
            ax.set_title(f'Distribution of {n} Points in Unit Square', fontsize=14, style='italic')

            # Display the plot in Streamlit
            st.pyplot(fig)

    with col2:
        if fig and ax:
            ax.scatter(transformed_grid_points[:, 0], transformed_grid_points[:, 1], color='b', s=10, alpha=0.5)
            plt.xlim(-1.5, 1.5)
            plt.ylim(-1.5, 1.5)
            plt.axhline(0, color='black', linewidth=0.5)
            plt.axvline(0, color='black', linewidth=0.5)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title(f'Cartesian grid space in background of Unit Square', fontsize=14, style='italic')
            st.pyplot(fig)

    # Second Row (Split into 2 columns)
#col3, col4 = st.columns([1, 1])
    import pandas as pd
    import os
    import plotly.express as px

    col3, col4 = st.columns([1, 2])

    # Relative path to the file
    file_path = os.path.join(os.path.dirname(__file__), "data", "transformations_new_results.xlsx")
    df = pd.read_excel(file_path, index_col=None)

    # Dynamically filter column B (Inside Points) based on N
    if "Inside Points" in df.columns:
        filtered_df = df[df["Inside Points"] == n]
    else:
        st.warning("The column 'Inside Points' does not exist in the uploaded Excel file.")
        filtered_df = df

    # Define the column to hide
    column_to_hide = "Inside Points"

    # Check if the column exists in the DataFrame
    if column_to_hide in filtered_df.columns:
        # Create a new DataFrame without the column
        filtered_df_to_display = filtered_df.drop(columns=[column_to_hide])
    else:
        # If the column doesn't exist, use the original DataFrame
        filtered_df_to_display = filtered_df
    # Left column: Table display
    # Display the table without the hidden column

    #with st.columns([1, 20])[1]:
    #    st.write(f'Additional transformations that give same {n} inside the Unit Square:',
        #         filtered_df_to_display.to_html(index=False), unsafe_allow_html=True)

    # Right column: 4D scatterplot matrix visualization
    with st.columns([1, 20])[1]:
        columns_to_plot = ["Scaling Factor", "Rotation Degree", "Translation X", "Translation Y"]

        # Check for missing columns
        missing_columns = [col for col in columns_to_plot if col not in filtered_df.columns]
        if missing_columns:
            st.warning(f"Missing required columns for visualization: {', '.join(missing_columns)}")
        else:
            st.write("### 4D Scatterplot Matrix")

            # Ensure the correct column name is used for coloring
            color_column = "Scaling Factor" if "Scaling Factor" in filtered_df.columns else None

            fig = px.scatter_matrix(
                filtered_df,
                dimensions=columns_to_plot,
                color=color_column,  # Corrected column name
                title=f"Transformations of {n} Scatterplot Matrix",
                labels={col: col.replace(" ", " ") for col in columns_to_plot},  # Adjusted space handling
                height=700,
                width=900
            )
            st.plotly_chart(fig, use_container_width=True)
