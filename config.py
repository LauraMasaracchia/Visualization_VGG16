# default input dimensions
INPUT_DIMENSIONS = (1, 480, 360, 3)

# VGG_16 specifications
DEFAULT_NETWORK_NAME = "VGG16"

VGG16_LAYER_NAMES_LIST = ['block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1',
                          'block2_conv2', 'block2_pool', 'block3_conv1', 'block3_conv2',
                          'block3_conv3', 'block3_pool', 'block4_conv1', 'block4_conv2',
                          'block4_conv3', 'block4_pool', 'block5_conv1', 'block5_conv2',
                          'block5_conv3', 'block5_pool']

VGG16_WEIGHT_DIMENSIONS_LIST = [(3, 3, 3, 64), (3, 3, 64, 64), (), (3, 3, 64, 128), (3, 3, 128, 128), (),
                                (3, 3, 128, 256), (3, 3, 256, 256), (3, 3, 256, 256), (), (3, 3, 256, 512),
                                (3, 3, 512, 512), (3, 3, 512, 512), (), (3, 3, 512, 512), (3, 3, 512, 512),
                                (3, 3, 512, 512), ()]

# path to trained weights of VGG_16 (without the classifier on top)
VGG16_weights_file_name = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
TRAINED_VGG16_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
