import math
import numpy as np
from matplotlib import pyplot as plt


def show_matrices(matrices: list, labels: list):
    fig = plt.figure()
    max_value = np.max(matrices)

    if all(matrix.shape[1] == 3 for matrix in matrices):
        axis = fig.add_subplot(111, projection='3d')
        for i, matrix in enumerate(matrices):
            xs, ys, zs = matrix[:, 0], matrix[:, 1], matrix[:, 2]
            axis.plot(xs, ys, zs, label=labels[i])

        axis.set_xlim(-max_value, max_value)
        axis.set_ylim(-max_value, max_value)
        axis.set_zlim(-max_value, max_value)
    elif all(matrix.shape[1] == 2 for matrix in matrices):
        axis = fig.add_subplot(111)
        for i, matrix in enumerate(matrices):
            xs, ys = matrix[:, 0], matrix[:, 1]
            axis.plot(xs, ys, label=labels[i])
        plt.xlim(-max_value, max_value)
        plt.ylim(-max_value, max_value)
    else:
        raise ValueError("Impossible to show this matrices.")

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

    show_matrices([original_matrix, transformed_matrix], ['Original', operation_label])


def scale_matrix(original_matrix: np.array, scales: list):
    trans_matrix = np.diag(scales)
    transform_by_matrix(original_matrix, trans_matrix, f"Scaled by {scales}")


def rotate_matrix(original_matrix: np.array, angle: float, plane=(0, 1)):
    angle_rad = math.radians(angle)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    trans_matrix = np.eye(original_matrix.shape[1])
    x, y = plane
    trans_matrix[[x, x, y, y], [x, y, x, y]] = [cos_a, -sin_a, sin_a, cos_a]
    transform_by_matrix(original_matrix, trans_matrix, f"Rotated by {angle}º in {plane} plane")


def reflect_matrix(original_matrix: np.array, axes: list):
    axes_names = np.array(['x', 'y', 'z'])
    reflect_vector = [-1 if axis else 1 for axis in axes]
    if len(axes) == 2:
        axes.append(False)
    trans_matrix = np.diag(reflect_vector)
    transform_by_matrix(original_matrix, trans_matrix, f"Reflected across {axes_names[axes]}")


def angle_matrix(original_matrix: np.array, k: float, fixed_axis: int, variable_axis: int):
    trans_matrix = np.eye(original_matrix.shape[1])
    trans_matrix[variable_axis, fixed_axis] = k
    transform_by_matrix(original_matrix, trans_matrix, f"Angled by {k} along {fixed_axis} axis")


batman = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]])
rotate_matrix(batman, 45)
scale_matrix(batman, [2, 2])
reflect_matrix(batman, [False, True])
angle_matrix(batman, 2, 0, 1)

trans_matrix = np.array([[0, 1], [1, 0]])
transform_by_matrix(batman, trans_matrix)

cube = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]])
show_matrices([cube], ['Original'])

rotate_matrix(cube, 45, (0, 1))
rotate_matrix(cube, 45, (1, 2))
rotate_matrix(cube, 45, (0, 2))
scale_matrix(cube, [1, 2, 1])
reflect_matrix(cube, [True, False, False])
angle_matrix(cube, 1, 2, 1)
