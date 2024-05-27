import numpy as np
from matplotlib import pyplot as plt


def transform(original_matrix, transform_matrix):
    original_matrix = np.array(original_matrix)
    transform_matrix = np.array(transform_matrix)

    if original_matrix.shape[1] == transform_matrix.shape[0]:
        transformed_matrix = np.dot(original_matrix, transform_matrix)
    elif original_matrix.shape[0] == transform_matrix.shape[1]:
        transformed_matrix = np.dot(transform_matrix, original_matrix)
    else:
        raise ValueError(
            "The number of columns in the original matrix must match the number of rows in the transform matrix.")

    fig, ax = plt.subplots()
    colors = ['b', 'g']

    for i in range(len(original_matrix)):
        ax.quiver(0, 0, original_matrix[i, 0], original_matrix[i, 1], angles='xy', scale_units='xy', scale=1,
                  color=colors[0], width=0.005, label=f"Original Vec {i + 1}")

    for i in range(len(transformed_matrix)):
        ax.quiver(0, 0, transformed_matrix[i, 0], transformed_matrix[i, 1], angles='xy', scale_units='xy', scale=1,
                  color=colors[1], width=0.005, label=f"Transformed Vec {i + 1}")

    max_value = np.max(transformed_matrix)
    plt.xlim(-max_value, max_value)
    plt.ylim(-max_value, max_value)
    ax.set_aspect('equal')
    plt.grid(True)
    # plt.legend()
    plt.show()


batman = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]])
transform_matrix = np.array([[1, 0], [1, 1]])
transform(batman, transform_matrix)
