import numpy as np
import matplotlib.pyplot as plt

# Define the depth of the tree
n = 7
# Calculate the total number of nodes in the tree using a geometric progression formula
n = (5**(n+1) - 1) // 4

# Initialize a 2D array to hold the tree's structure (coordinates for each node)
# The first index represents x and y coordinates, the second represents each node, and the third represents four control points
X = np.zeros((2, n, 4))  # 2 for x and y coordinates

# Define the initial branch (base of the tree) with four control points
X[0,0] = np.array([-2, 2, 1, -1])  # x-coordinates
X[1,0] = np.array([-14, -14, 5, 5])  # y-coordinates

# Scale down the tree by a factor of 2
X /= 2

# Index to keep track of saved branches in the array
save_ind = 1

# Function to generate a gradient of colors from brown to green (or pink for blossoms)
def generate_colors(n):
    """
    Generates an array of colors interpolated from brown to pink.

    Parameters:
    n (int): Number of colors to generate.

    Returns:
    np.ndarray: Array of colors in RGB format.
    """
    brown = np.array([0.104, 0.059, 0.016])  # Brown for branches
    pink = np.array([1, 183/255, 197/255])  # Pink for cherry blossoms

    # Create a smooth color transition using hyperbolic tangent
    t = np.tanh(np.linspace(0, n, n) / n**0.75)

    # Interpolate between brown and pink based on the value of t
    colors = np.outer(1 - t, brown) + np.outer(t, pink)
    
    return colors

# Function to generate a 2D rotation matrix for a given angle
def gyro(ang):
    cos = np.cos(ang)
    sin = np.sin(ang)
    return np.array([[cos, -sin], [sin, cos]])

# Function to reduce the scale factor over distance
def reduction_factor(x):
    return (2 - 1) * np.tanh(-x / 1000) + 2

# Function to apply translation to each row of a matrix
def translate(matrix, row_values):
    """
    Add specific values to each row of the matrix.

    Parameters:
    matrix (np.ndarray): The input matrix.
    row_values (list or np.ndarray): A list or array of values to add to each row.

    Returns:
    np.ndarray: A new matrix with the specified values added to each row.
    """
    row_values = np.array(row_values)[:, np.newaxis]  # Convert row_values to a column vector
    result = matrix + row_values  # Add row values to the matrix using broadcasting
    return result

# Generate rotation matrices for branch orientation
matrix_1 = gyro(1.8 * np.pi / 10)
matrix_2 = gyro(-1.5 * np.pi / 8)
matrix_3 = gyro(-1.8 * np.pi / 10)

# Loop through each branch level, creating and translating sub-branches
for current_ind in range(n // 5):
    # Create the main branch for the current level and translate it upwards
    branch = translate(X[:, current_ind] / 2, [0, 5.9])
    
    # Create sub-branches with different rotations and translations
    sub_branch2 = translate(np.dot(matrix_1, translate(branch, [0, -2.5])), [-0.1, 3.5])
    sub_branch3 = translate(np.dot(matrix_2, translate(branch, [0, -2.5])), [0.35, -3])
    sub_branch4 = translate(np.dot(matrix_1, translate(branch, [0, -2.5])), [-0.25, 0])
    sub_branch5 = translate(np.dot(matrix_3, translate(branch, [0, -2.5])), [0.2, 1])
    
    # Save the generated branches into the array X
    X[:, save_ind, :] = branch
    X[:, save_ind+1, :] = sub_branch2
    X[:, save_ind+2, :] = sub_branch3
    X[:, save_ind+3, :] = sub_branch4
    X[:, save_ind+4, :] = sub_branch5
    save_ind += 5  # Move the index to the next set of branches

# Save the generated tree structure into a file
np.save('tree.npy', X)

# Create a figure and an axis for plotting
fig, ax = plt.subplots()

# Set axis limits based on the data
ax.set_xlim(np.min(X[0]), np.max(X[0]))
ax.set_ylim(np.min([X[1]]), np.max(X[1]))
ax.axis('off')  # Hide the axis

# Generate colors for the branches
colors = generate_colors(n)

# Plot each branch as a polygon and add it to the plot
for i, color in enumerate(colors):
    polygon = plt.Polygon(X[:, i].T, closed=True, fill=True, edgecolor=None, facecolor=color)
    ax.add_patch(polygon)

# Save the final tree as a PDF
fig.savefig('cerezo.pdf', format='pdf')

# Display the plot
plt.show()
