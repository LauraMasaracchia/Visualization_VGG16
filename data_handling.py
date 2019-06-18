import cv2
import numpy as np


def Plot_image(array_image, cv2_colormap=None):
    image = array_image[0, :, :, :].astype('uint8')
    image = np.swapaxes(image, 0, 1)
    if cv2_colormap is None:
        cv2.imshow("figure", image)
    else:
        cv2.imshow("pixel_space_feature", cv2.applyColorMap(image, cv2_colormap))
        cv2.waitKey(0)



def Read_image(file_path):
    starting_image = cv2.imread(file_path)
    if starting_image is None:
        raise AssertionError("no valid path")
    else:
        array_img = np.asarray(starting_image)
        return array_img


def Visualization_Tool_fit_image(array_img, output_dim):

    batch_size = output_dim[0]
    x_dim = output_dim[1]
    y_dim = output_dim[2]
    channels = output_dim[3]

    img = np.swapaxes(array_img, axis1=0, axis2=1) # all images are loaded with swapped axes.
    real_shape = img.shape
    image_reshaped = np.zeros(shape=(batch_size, x_dim, y_dim, channels))

    # crop center part of the image if the spatial dimensions exceed the desired output dimensions
    # center the image and fill the rest with zeros if the original image is smaller than the desired output dimensions

    # case horizontal dimension of the image is bigger than the desired horizontal shape
    if real_shape[0] > x_dim:
        x_margin = int((real_shape[0] - x_dim)/2)
        x_start = x_margin
        x_stop = real_shape[0] - x_margin - 1 * (real_shape[0] % 2)
        case_x = 1

    # case the horizontal dimension is smaller than the desired shape
    else:
        case_x = 0
        x_margin = int((x_dim - real_shape[0]) / 2)
        x_start = x_margin
        x_stop = x_dim - x_margin + 1 * (real_shape[0] % 2)

    # case vertical dimension of the image is bigger than the desired vertical shape
    if real_shape[1] > y_dim:
        y_margin = int((real_shape[1] - y_dim) / 2)
        y_start = y_margin
        y_stop = real_shape[1] - y_margin - 1 * (real_shape[1] % 2)
        case_y = 1

    # case the vertical dimension is smaller than the desired shape
    else:
        y_margin = int((y_dim - real_shape[1]) / 2)
        y_start = y_margin
        y_stop = y_dim - y_margin + 1 * (real_shape[1] % 2)
        case_y = 0

    # fit the image with the right case
    if case_x == 1 and case_y == 1: # x and y dimensions bigger than the desired shape: crop the image
        image_reshaped[0, :, :, :] = img[x_start:x_stop, y_start:y_stop, :]
    elif case_x == 1 and case_y == 0: # x dim bigger, y dim smaller than the desired shape
        image_reshaped[0, :, y_start:y_stop, :] = img[x_start:x_stop, :, :]
    elif case_y == 1 and case_x == 0: # y dim bigger, x dim smaller than the desired shape
        image_reshaped[0, x_start:x_stop, :, :] = img[:, y_start:y_stop, :]
    else: # original image smaller than the desired shape: center and fit in a numpy array of zeros
        image_reshaped[0, x_start:x_stop, y_start:y_stop, :] = img[:, :, :]

    return image_reshaped
