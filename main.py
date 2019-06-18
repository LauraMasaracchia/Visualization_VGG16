from Visualization_VGG16.config import *
from Visualization_VGG16.utils.data_handling import *
from Visualization_VGG16.Visualizer_VGG16 import *


# change according to your desire
LAYER_NUMBER = 'block3_conv2'
FEATURE_MAP_NUMBER = 35
IMAGE_FILE_PATH = 'example_data/cat.jpg'
#IMAGE_FILE_PATH = 'example_data/car.jpg'


# read image
raw_image = Read_image(IMAGE_FILE_PATH)

# reshape image to fit the model
image = Visualization_Tool_fit_image(raw_image, INPUT_DIMENSIONS)

# create the model ready to visualize one layer and one feature map
Visualizer = VGG16_Visualizer(VGG16_LAYER_NAMES_LIST, VGG16_WEIGHT_DIMENSIONS_LIST, TRAINED_VGG16_WEIGHTS_PATH, INPUT_DIMENSIONS)

ad_hoc_model = Visualizer.Generate_Model(LAYER_NUMBER, FEATURE_MAP_NUMBER)

# get the feature map down to pixel space
pixel_space_feature = ad_hoc_model.predict(image)

# plot
Plot_image(array_image=image)
Plot_image(array_image=pixel_space_feature, cv2_colormap=cv2.COLORMAP_HOT)
