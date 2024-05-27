import numpy as np
from matplotlib import pyplot as plt


def add_polygon(axis, matrix, color, label):
    axis.add_patch(plt.Polygon(matrix, closed=True, fill=None, edgecolor=color, label=label))


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

    axis = plt.subplots()[1]

    add_polygon(axis, original_matrix, 'b', 'Original')
    add_polygon(axis, transformed_matrix, 'r', 'Transformed')

    all_points = np.concatenate((original_matrix, transformed_matrix))
    max_value = np.max(np.abs(all_points))
    plt.xlim(-max_value, max_value)
    plt.ylim(-max_value, max_value)

    plt.grid(True)
    plt.legend()
    plt.show()


batman = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]])
transform_matrix = np.array([[1, 0], [1, 1]])
transform(batman, transform_matrix)
