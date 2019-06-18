""" Function that transforms one hot encoded into grayscale picture.
 Inputs:
  one_hot_softmax : ONE image(a numpy array) encoded with a probability distribution over classes
  - with shape (x_dim, y_dim, nbr_classes).
  class_list : a list of numbers - of length nbr_classes - to represent each class encoding in gray scale.
  (optionally: a probability threshold above which the probability is considered enough - generally consistently above
  random chance level - to determine a preference of the network.

 Outputs:
  a numpy arrays of dimensions (x_dim, y_dim) - a grayscale image.
"""

import numpy as np


def convert_one_hot_to_instance(one_hot_softmax, class_list, prob_threshold=None):
    # check right shape and number of classes
    nbr_classes = len(class_list)
    if len(one_hot_softmax.shape) != 3:
        raise AssertionError("Input must have shape (x, y, nbr_classes)")

    nbr_channels = one_hot_softmax.shape[2]
    x_dim = one_hot_softmax.shape[0]
    y_dim = one_hot_softmax.shape[1]

    if nbr_channels != nbr_classes:
        raise AssertionError("Number of classes listed and encoded must be the same!")

    # set threshold probability to determine a class is actually consistently preferred over the others
    if prob_threshold is None:
        prob_threshold = (1 / nbr_classes)

    instances_grayscale = np.zeros(shape=(x_dim, y_dim))

    # For every pixel check that the highest probability is higher than prob_threshold.
    # If not, then it will be encoded as undecided class. 0
    # cannot set to 0 everything that is below prob_threshold
    for i in range(x_dim):
        for j in range(y_dim):
            m = np.argmax(one_hot_softmax[i, j, :])
            if one_hot_softmax[i, j, m] > prob_threshold:
                instances_grayscale[i, j] = class_list[m]

    return instances_grayscale





def combine_image_prediction(prediction, image, alpha=None):
    """ takes as input ONE prediction in form of GRAYSCALE image and the image input.
    (The predictions in form of images are created by the postprocessor convert_one_hot_to_instance, that uses the
    visualization tool softmax_to_instances_converter.)

     alpha: parameter to indicate proportion of value of the prediction pixel on the final image. By Default: 0.5
    """

    # check that prediction is grayscale
    if len(prediction.shape) != 2:
        raise AssertionError("Prediction has to be one grayscale image with dimension (x,y)")
    if prediction.shape[0] != image.shape[0] or prediction.shape[1] != image.shape[1]:
        raise AssertionError("Prediction and Image must be of same x and y dimensions")
    if alpha is None:
        alpha = 0.5
    elif alpha > 1 or alpha < 0:
        raise AssertionError("alpha is a fraction. needs to be between 0 and 1")

    combined0 = prediction * alpha + image[:, :, 0] * (1. - alpha)
    combined1 = prediction * alpha + image[:, :, 1] * (1. - alpha)
    combined2 = prediction * alpha + image[:, :, 2] * (1. - alpha)
    image[:, :, 0] = combined0
    image[:, :, 1] = combined1
    image[:, :, 2] = combined2
    return image

