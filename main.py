import math
import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_matrices(matrices: list, labels: list, save=False):
    fig = plt.figure()
    max_value = np.max(matrices)
    min_value = np.min(matrices)

    if all(matrix.shape[1] == 3 for matrix in matrices):
        axis = fig.add_subplot(111, projection='3d')
        for i, matrix in enumerate(matrices):
            xs, ys, zs = matrix[:, 0], matrix[:, 1], matrix[:, 2]
            axis.plot(xs, ys, zs, label=labels[i])

        axis.set_xlabel('X')
        axis.set_ylabel('Y')
        axis.set_zlabel('Z')
        axis.set_xlim(min_value, max_value)
        axis.set_ylim(min_value, max_value)
        axis.set_zlim(min_value, max_value)
    elif all(matrix.shape[1] == 2 for matrix in matrices):
        axis = fig.add_subplot(111)
        for i, matrix in enumerate(matrices):
            xs, ys = matrix[:, 0], matrix[:, 1]
            axis.plot(xs, ys, label=labels[i])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(min_value, max_value)
        plt.ylim(min_value, max_value)
    else:
        raise ValueError("Impossible to show this matrices.")

    if not save:
        plt.grid(True)
        plt.legend()
    else:
        plt.savefig('plot.png')
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


def rotate_matrix(original_matrix: np.array, angle: float, axis='y'):
    angle_rad = math.radians(angle)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    if original_matrix.shape[1] == 2:
        trans_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    elif original_matrix.shape[1] == 3:
        match axis:
            case 'x':
                trans_matrix = np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]])
            case 'y':
                trans_matrix = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
            case 'z':
                trans_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    else:
        raise ValueError("Rotation only supports 2D or 3D matrices.")

    transform_by_matrix(original_matrix, trans_matrix, f"Rotated by {angle}º around axis {axis}")


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


def show_image(image, message):
    plt.imshow(image, cmap='gray')
    plt.show()


def rotate_image(image, angle):
    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (width, height))
    show_image(rotated, f"Rotated by {angle}º")


def scale_image(image, scale_x, scale_y):
    scaled = cv2.resize(image, None, fx=scale_x, fy=scale_y)
    show_image(scaled, f"Scaled by {scale_x}, {scale_y} (OpenCV)")


def reflect_image(image, axis):
    if axis == 'x':
        reflected = cv2.flip(image, 0)
    else:
        reflected = cv2.flip(image, 1)
    show_image(reflected, f"Reflected across {axis} (OpenCV)")


def angle_image(image, k, axis):
    (height, width) = image.shape[:2]
    if axis == 'x':
        matrix = np.float32([[1, k, 0], [0, 1, 0]])
    else:
        matrix = np.float32([[1, 0, 0], [k, 1, 0]])
    angled = cv2.warpAffine(image, matrix, (width, height))
    show_image(angled, f"Angled by {k} along {axis} axis (OpenCV)")


batman = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1.5, 0.5], [0, 0]])
rotate_matrix(batman, 45)
scale_matrix(batman, [2, 2])
reflect_matrix(batman, [False, True])
angle_matrix(batman, 2, 0, 1)

trans_matrix = np.array([[0, 1], [1, 0]])
transform_by_matrix(batman, trans_matrix, f"Transformed by {trans_matrix} matrix")

cube = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]])

rotate_matrix(cube, 45, 'x')
rotate_matrix(cube, 45, 'y')
rotate_matrix(cube, 45, 'z')
scale_matrix(cube, [1, 2, 1])
reflect_matrix(cube, [True, True, False])
angle_matrix(cube, 1, 2, 1)

show_matrices([batman], [''], True)
plot = cv2.imread('plot.png')
rotate_image(plot, 45)
scale_image(plot, 2, 2)
reflect_image(plot, 'y')
angle_image(plot, 0.5, 'x')
