import numpy as np
from matplotlib import pyplot as plt


def show_transformation(matrices: list, labels: list):
    axis = plt.subplots()[1]
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

    i = 0
    for matrix in matrices:
        axis.add_patch(plt.Polygon(matrix, closed=True, fill=None, edgecolor=colors[i], label=labels[i]))
        i += 1

    max_value = np.max(np.abs(matrices))
    plt.xlim(-max_value, max_value)
    plt.ylim(-max_value, max_value)

    plt.grid(True)
    plt.legend()
    plt.show()


def transform(original_matrix, transform_matrix):
    original_matrix = np.array(original_matrix)
    transform_matrix = np.array(transform_matrix)

    if original_matrix.shape[1] == transform_matrix.shape[0]:
        transformed_matrix = np.dot(original_matrix, transform_matrix)
    elif original_matrix.shape[0] == transform_matrix.shape[1]:
        transformed_matrix = np.dot(transform_matrix, original_matrix)
    else:
        raise ValueError("Impossible to multiply this matrices.")

    show_transformation([original_matrix, transformed_matrix], ['Original', 'Transformed'])


batman = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]])
transform_matrix = np.array([[1, 0], [1, 1]])
transform(batman, transform_matrix)
