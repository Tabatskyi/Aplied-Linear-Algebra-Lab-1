import math
import cv2
import numpy as np
from matplotlib import pyplot as plt


def plot_matrices(matrices, labels, save_path=None):
    dimension = matrices[0].shape[1]
    fig = plt.figure()
    max_value = np.max(matrices)
    min_value = np.min(matrices)
    if dimension == 3:
        axis = fig.add_subplot(111, projection='3d')
        set_labels = ('X', 'Y', 'Z')
    else:
        axis = fig.add_subplot(111)
        set_labels = ('X', 'Y')

    for matrix, label in zip(matrices, labels):
        if dimension == 3:
            axis.plot(matrix[:, 0], matrix[:, 1], matrix[:, 2], label=label)
            axis.set_xlim(min_value, max_value)
            axis.set_ylim(min_value, max_value)
            axis.set_zlim(min_value, max_value)
        else:
            axis.plot(matrix[:, 0], matrix[:, 1], label=label)
            plt.xlim(min_value, max_value)
            plt.ylim(min_value, max_value)

    axis.set_xlabel(set_labels[0])
    axis.set_ylabel(set_labels[1])
    if dimension == 3:
        axis.set_zlabel(set_labels[2])
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def apply_matrix_transformation(original, transformation, label):
    transformed = np.dot(original, transformation)
    plot_matrices([original, transformed], ['Original', label])


def create_rotation_matrix(angle, axis, dimension):
    angle_rad = math.radians(angle)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    if dimension == 2:
        return np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    elif dimension == 3:
        if axis == 'x':
            return np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]])
        elif axis == 'y':
            return np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
        elif axis == 'z':
            return np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        else:
            raise ValueError("Invalid axis for rotation.")
    else:
        raise ValueError("Rotation only supports 2D or 3D matrices.")


def rotate_matrix(matrix, angle, axis='z'):
    dim = matrix.shape[1]
    rotation_matrix = create_rotation_matrix(angle, axis, dim)
    apply_matrix_transformation(matrix, rotation_matrix, f"Rotated by {angle}º around {axis} axis")


def scale_matrix(matrix, scales):
    scale_matrix = np.diag(scales)
    apply_matrix_transformation(matrix, scale_matrix, f"Scaled by {scales}")


def reflect_matrix(matrix, axes):
    reflect_vector = [-1 if axis else 1 for axis in axes]
    reflection_matrix = np.diag(reflect_vector)
    axes_names = ['x', 'y', 'z'][:len(axes)]
    apply_matrix_transformation(matrix, reflection_matrix, f"Reflected across {axes_names}")


def angle_matrix(matrix, k, fixed_axis, variable_axis):
    transformation_matrix = np.eye(matrix.shape[1])
    transformation_matrix[variable_axis, fixed_axis] = k
    apply_matrix_transformation(matrix, transformation_matrix, f"Angled by {k} along {fixed_axis} axis")


def get_angle_matrix(matrix, k, axis):
    (height, width) = matrix.shape[:2]
    if axis == 'x':
        src_points = np.float32([[0, 0], [width, 0], [0, height]])
        dst_points = np.float32([[0, 0], [width, 0], [k * height, height]])
    else:
        src_points = np.float32([[0, 0], [width, 0], [0, height]])
        dst_points = np.float32([[0, 0], [width, k * width], [0, height]])

    return cv2.getAffineTransform(src_points, dst_points)


def transform_image(image, matrix):
    (height, width) = image.shape[:2]
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def rotate_image(image, angle):
    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return transform_image(image, rotation_matrix)


def scale_image(image, scale_x, scale_y):
    height, width = image.shape[:2]
    new_width = int(width * scale_x)
    new_height = int(height * scale_y)
    return cv2.resize(image, (new_width, new_height))


def reflect_image(image, axis):
    if axis == 'x':
        return cv2.flip(image, 0)
    elif axis == 'y':
        return cv2.flip(image, 1)
    else:
        raise ValueError("Axis must be 'x' or 'y'.")


def angle_image(image, k, axis='x'):
    angle_matrix = get_angle_matrix(image, k, axis)
    return transform_image(image, angle_matrix)


def show_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


def rotate_matrix_opencv(image, angle, scale=1.0):
    matrix = cv2.getRotationMatrix2D((0, 0), angle, scale)
    return cv2.transform(np.array([image]), matrix)[0]


def scale_matrix_opencv(image, scale):
    return rotate_matrix_opencv(image, 0, scale)


def reflect_matrix_opencv(image, axis):
    if axis == 'x':
        return cv2.flip(image, 0)
    elif axis == 'y':
        return cv2.flip(image, 1)
    else:
        raise ValueError("Axis must be 'x' or 'y'.")


def angle_matrix_opencv(matrix, k, axis):
    angle_matrix = get_angle_matrix(matrix, k, axis)
    return cv2.transform(np.array([matrix]), angle_matrix)[0]


batman = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1.5, 0.5], [0, 0]])
rotate_matrix(batman, 45)
scale_matrix(batman, [2, 2])
reflect_matrix(batman, [False, True])
angle_matrix(batman, 2, 0, 1)

trans_matrix = np.array([[0, 1], [1, 0]])
apply_matrix_transformation(batman, trans_matrix, f"Transformed by {trans_matrix} matrix")

cube = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]])

rotate_matrix(cube, 45, 'x')
rotate_matrix(cube, 45, 'y')
rotate_matrix(cube, 45, 'z')
scale_matrix(cube, [1, 2, 1])
reflect_matrix(cube, [True, True, False])
angle_matrix(cube, 1, 2, 1)

# plot = cv2.imread('plot.png')
rotated = rotate_matrix_opencv(batman, 45)
plot_matrices([batman, rotated], ['Original', f"Rotated by 45º (OpenCV)"])

scaled = scale_matrix_opencv(batman, 2)
plot_matrices([batman, scaled], ['Original', f"Scaled by [2, 2] (OpenCV)"])

reflected = reflect_matrix_opencv(batman, 'y')
plot_matrices([batman, reflected], ['Original', f"Reflected by y (OpenCV)"])

angled = angle_matrix_opencv(batman, 1, 'x')
plot_matrices([batman, angled], ['Original', f"Scaled by [2, 2] (OpenCV)"])

rick = cv2.imread('rick.png')
angled_rick = angle_image(rick, 0.5, 'x')
reflected_rick = reflect_image(angled_rick, 'y')
rotated_rick = rotate_image(reflected_rick, 90)
show_image(rotated_rick)

rick2 = cv2.imread('rick.png')
rotated_rick2 = rotate_image(rick2, 90)
reflected_rick2 = reflect_image(rotated_rick2, 'y')
angled_rick2 = angle_image(reflected_rick2, 0.5, 'x')

show_image(angled_rick2)
