from Visualization_VGG16.utils.VGG16_Visualizer_functions import *
from Visualization_VGG16.config import *


class VGG16_Visualizer():
    def __init__(self, layer_names_list, layer_weights_dimensions_list, trained_weights_path, input_dimension):
        self.layer_names_list = layer_names_list
        self.layer_weights_dimensions_list = layer_weights_dimensions_list
        self.trained_weights_path = trained_weights_path
        self.input_dimension = input_dimension

    def Generate_Model(self, layer_name, channel):
        # check input validity
        if layer_name not in self.layer_names_list:
            raise AssertionError("Invalid layer name! (check spelling)")

        # corresponding layer number
        selected_layer_nbr = self.layer_names_list.index(layer_name)

        # get shape of weights to modify
        selected_weight_shape = Get_Shape_of_Weights_to_Modify(selected_layer_nbr, self.layer_weights_dimensions_list)

        model = Build_end2end_VGG16_VT_Model(selected_layer_nbr, self.input_dimension)

        # Compose ad-hoc weights from pre-trained VGG16
        trained_encoder_weights = Retrieve_Encoder_weights(layer_name, self.trained_weights_path)
        model_full_weights = Compose_VGG16_VT_weights(trained_encoder_weights)

        # modify the weights depending on the channel we want to visualize
        model_full_weights = Modify_Selected_Channel_Weights(weights=model_full_weights,
                                                             selected_weight_shape=selected_weight_shape,
                                                             channel=channel)
        # set the modified weights
        model.set_weights(model_full_weights)

        return model

