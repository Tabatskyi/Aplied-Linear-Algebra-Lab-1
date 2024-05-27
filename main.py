import math
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


def transform_by_matrix(original_matrix: np.array, transform_matrix: np.array, operation_label='Transformed'):
    if original_matrix.shape[1] == transform_matrix.shape[0]:
        transformed_matrix = np.dot(original_matrix, transform_matrix)
    elif original_matrix.shape[0] == transform_matrix.shape[1]:
        transformed_matrix = np.dot(transform_matrix, original_matrix)
    else:
        raise ValueError("Impossible to multiply this matrices.")

    show_transformation([original_matrix, transformed_matrix], ['Original', operation_label])


def rotate_matrix(original_matrix: np.array, angle: float):
    angle_rad = math.radians(angle)
    trans_matrix = np.array([[math.cos(angle_rad), -math.sin(angle_rad)], [math.sin(angle_rad), math.cos(angle_rad)]])
    transform_by_matrix(original_matrix, trans_matrix, f"Rotated by {angle}º")


def scale_matrix(original_matrix: np.array, scale_x: float, scale_y: float):
    trans_matrix = np.array([[scale_x, 0], [0, scale_y]])
    transform_by_matrix(original_matrix, trans_matrix, f"Scaled by {scale_x} on x and {scale_y} on y")


def reflect_matrix(original_matrix: np.array, axis: str):
    if axis == 'x':
        trans_matrix = np.array([[1, 0], [0, -1]])
    elif axis == 'y':
        trans_matrix = np.array([[-1, 0], [0, 1]])
    transform_by_matrix(original_matrix, trans_matrix, f"Reflected by {axis}")


def angle_matrix(original_matrix: np.array, k: float, axis: str):
    if axis == 'x':
        trans_matrix = np.array([[1, k], [0, 1]])
    else:
        trans_matrix = np.array([[1, 0], [k, 1]])
    transform_by_matrix(original_matrix, trans_matrix, f"Angled by {axis} with {k} coefficient")


batman = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]])
rotate_matrix(batman, 45)
scale_matrix(batman, 2, 2)
reflect_matrix(batman, 'x')
angle_matrix(batman, 2, 'y')

trans_matrix = np.array([[0, 1], [1, 0]])
transform_by_matrix(batman, trans_matrix)
