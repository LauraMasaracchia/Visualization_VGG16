from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import h5py
from keras.utils.data_utils import get_file
from keras.layers.convolutional import Conv2D, Deconv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.layers import Input
from Visualization_VGG16.utils.custom_layers import *



""" function generating model of visualization tool VGG_16 based.
 core function:
 The model of the encoder is built up to the layer number fed as input.
 the layers of the encoder are based on the structure of VGG16.
 VGG16 (without the top classifier) has 18 layers.
 the decoder part is mirroring the built encoder (from the inner to the most shallow layers)
 using upsampling (DePooling) instead of downsampling (MaxPooling)
 and convolutions with transposed weights (Deconvolution) instead of the normal convolutions.
 The main difference with the original VGG16 inner structure is that a MASK pooling layer is inserted
 before every max pooling extracting a mask with the indices of top values.
 These masks are then used by the decoder part to restore only the important locations with an upsampling layer. """

def Build_end2end_VGG16_VT_Model(selected_layer_nbr, input_batch_shape):

    img_input = Input(batch_shape=input_batch_shape)

    # encoder part
    masks = {}
    # Block 1
    if selected_layer_nbr >= 0:
        x = Conv2D(64, (3, 3), activation='relu', padding='same', use_bias=False, name='block1_conv1')(img_input)
    if selected_layer_nbr >= 1:
        x = Conv2D(64, (3, 3), activation='relu', padding='same', use_bias=False, name='block1_conv2')(x)
    if selected_layer_nbr >= 2:
        mask_1 = MaskPooling2D(padding='SAME')(x)
        masks['block1_mask'] = mask_1
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    if selected_layer_nbr >= 3:
        x = Conv2D(128, (3, 3), activation='relu', padding='same', use_bias=False, name='block2_conv1')(x)
    if selected_layer_nbr >= 4:
        x = Conv2D(128, (3, 3), activation='relu', padding='same', use_bias=False, name='block2_conv2')(x)
    if selected_layer_nbr >= 5:
        mask_2 = MaskPooling2D(padding='SAME')(x)
        masks['block2_mask'] = mask_2
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
    if selected_layer_nbr >= 6:
        x = Conv2D(256, (3, 3), activation='relu', padding='same', use_bias=False, name='block3_conv1')(x)
    if selected_layer_nbr >= 7:
        x = Conv2D(256, (3, 3), activation='relu', padding='same', use_bias=False, name='block3_conv2')(x)
    if selected_layer_nbr >= 8:
        x = Conv2D(256, (3, 3), activation='relu', padding='same', use_bias=False, name='block3_conv3')(x)
    if selected_layer_nbr >= 9:
        mask_3 = MaskPooling2D(padding='SAME')(x)
        masks['block3_mask'] = mask_3
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
    if selected_layer_nbr >= 10:
        x = Conv2D(512, (3, 3), activation='relu', padding='same', use_bias=False, name='block4_conv1')(x)
    if selected_layer_nbr >= 11:
        x = Conv2D(512, (3, 3), activation='relu', padding='same', use_bias=False, name='block4_conv2')(x)
    if selected_layer_nbr >= 12:
        x = Conv2D(512, (3, 3), activation='relu', padding='same', use_bias=False, name='block4_conv3')(x)
    if selected_layer_nbr >= 13:
        mask_4 = MaskPooling2D(padding='SAME')(x)
        masks['block4_mask'] = mask_4
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
    if selected_layer_nbr >= 14:
        x = Conv2D(512, (3, 3), activation='relu', padding='same', use_bias=False, name='block5_conv1')(x)
    if selected_layer_nbr >= 15:
        x = Conv2D(512, (3, 3), activation='relu', padding='same', use_bias=False, name='block5_conv2')(x)
    if selected_layer_nbr >= 16:
        x = Conv2D(512, (3, 3), activation='relu', padding='same', use_bias=False, name='block5_conv3')(x)
    if selected_layer_nbr >= 17:
        mask_5 = MaskPooling2D(padding='SAME')(x)
        masks['block5_mask'] = mask_5
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # decoder part

    # Block5
    if selected_layer_nbr == 17:
        x = DePooling2D()([x, mask_5])
    if selected_layer_nbr >= 16:
        x = Deconv2D(512, (3, 3), activation='relu', padding='same', use_bias=False, name='block5_deconv3')(x)
    if selected_layer_nbr >= 15:
        x = Deconv2D(512, (3, 3), activation='relu', padding='same', use_bias=False, name='block5_deconv2')(x)
    if selected_layer_nbr >= 14:
        x = Deconv2D(512, (3, 3), activation='relu', padding='same', use_bias=False, name='block5_deconv1')(x)

    # Block 4
    if selected_layer_nbr >= 13:
        x = DePooling2D()([x, mask_4])
    if selected_layer_nbr >= 12:
        x = Deconv2D(512, (3, 3), activation='relu', padding='same', use_bias=False, name='block4_deconv3')(x)
    if selected_layer_nbr >= 11:
        x = Deconv2D(512, (3, 3), activation='relu', padding='same', use_bias=False, name='block4_deconv2')(x)
    if selected_layer_nbr >= 10:
        x = Deconv2D(256, (3, 3), activation='relu', padding='same', use_bias=False, name='block4_deconv1')(x)

    # Block 3
    if selected_layer_nbr >= 9:
        x = DePooling2D()([x, mask_3])
    if selected_layer_nbr >= 8:
        x = Deconv2D(256, (3, 3), activation='relu', padding='same', use_bias=False, name='block3_deconv3')(x)
    if selected_layer_nbr >= 7:
        x = Deconv2D(256, (3, 3), activation='relu', padding='same', use_bias=False, name='block3_deconv2')(x)
    if selected_layer_nbr >= 6:
        x = Deconv2D(128, (3, 3), activation='relu', padding='same', use_bias=False, name='block3_deconv1')(x)

    # Block 2
    if selected_layer_nbr >= 5:
        x = DePooling2D()([x, mask_2])
    if selected_layer_nbr >= 4:
        x = Deconv2D(128, (3, 3), activation='relu', padding='same', use_bias=False, name='block2_deconv2')(x)
    if selected_layer_nbr >= 3:
        x = Deconv2D(64, (3, 3), activation='relu', padding='same', use_bias=False, name='block2_deconv1')(x)

    # Block 1
    if selected_layer_nbr >= 2:
        x = DePooling2D()([x, mask_1])
    if selected_layer_nbr >= 1:
        x = Deconv2D(64, (3, 3), activation='relu', padding='same', use_bias=False, name='block1_deconv2')(x)
    if selected_layer_nbr >= 0:
        x = Deconv2D(3, (3, 3), activation='relu', padding='same', use_bias=False, name='block1_deconv1')(x)

    predictions = x

    # build the model
    model = Model(inputs=img_input, outputs=predictions)

    return model


""" def Retrieve_Encoder_weights(selected_layer_name: string - layer to visualize, 
                                 file_path: string - path to trained weights):: returns list of numpy array pairs"""
def Retrieve_Encoder_weights(selected_layer_name, file_path):
    file_name = file_path.split("/")[-1]
    file = get_file(file_name, file_path, cache_subdir='models')
    f = h5py.File(file, mode='r')
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    filtered_layer_names = []
    for name in layer_names:
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        if weight_names:
            filtered_layer_names.append(name)

        if name == selected_layer_name:
            break
    layer_names = filtered_layer_names
    weights_arrays_list = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]
        array_weight = np.array(weight_values[0])
        array_biases = np.array(weight_values[1])
        weight_bias_array_list = [array_weight, array_biases]

        weights_arrays_list.append(weight_bias_array_list)

    return weights_arrays_list


""" def Compose_VGG16_VT_weights(weights_arrays_list: list of numpy arrays pairs):: returns list of numpy arrays
Takes out the biases and adds to the encoder weights themselves in a mirror structure for the decoding part"""
def Compose_VGG16_VT_weights(weights_arrays_list):
    #retrieve VGG16 weights WITHOUT BIASES up to layer selected_layer_name
    encoder_length = len(weights_arrays_list)
    weights_to_share = []
    for i in range(encoder_length):
        weights_to_share.append(weights_arrays_list[i][0])
    for i in range(encoder_length - 1, -1, -1):
        weights_to_share.append(weights_arrays_list[i][0])

    return weights_to_share


def Modify_Selected_Channel_Weights(weights, selected_weight_shape, channel):
        # get max number of channels to avoid input errors
    max_channel_nbr = selected_weight_shape[-1] - 1

    if channel is not 'all':
        if channel > max_channel_nbr:
            channel = max_channel_nbr

        weights_modified = np.zeros(shape=selected_weight_shape)
        weights_modified[:, :, :, channel] = weights[int(len(weights) / 2)][:, :, :, channel]
        weights[int(len(weights) / 2)] = weights_modified

    return weights


def Get_Shape_of_Weights_to_Modify(selected_layer_nbr, layer_weights_dimensions_list):
    # layer_weights_dimensions_list is a list of tuples with dimensions of weights in ALL VGG16 layers

    if not layer_weights_dimensions_list[selected_layer_nbr]:  # empty tuple if max_pooling layer
        selected_weight_shape = layer_weights_dimensions_list[selected_layer_nbr - 1]
    else:
        selected_weight_shape = layer_weights_dimensions_list[selected_layer_nbr]
    return selected_weight_shape
