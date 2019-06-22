# Visualization_VGG16
Visualization tool, encoder-decoder structure, to project down to pixel level the inner activations of VGG16

Following the work done by M. D. Zeiler and R. Fergus, in their paper "Visualizing and Understanding Convolutional Networks", ECCV 2014 (https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf), 

I developed a visualization tool able to project down to pixel space the activations of any given layer and feature maps of a VGG16. 

The visualization tool works as follows: an image is fed to an encoder dynamically build on the basis of VGG16 (pre-trained on the imagenet dataset). This first part is built up to the layer we desire to visualize. 

Then, a mirror network is built to project down to the pixel space the activations of the selected layer. 

The mirror network uses upsampling instead of downsampling and deconvolutions (convolution with transposed weights) in the place of convolutions. The overall structure in this way reminds that of any autoencoder. 

The key point is to make the deconvolutional layers use the EXACT SAME WEIGHTS learned by the convolutional layers to perform the classification task. 

Another important point is to store the indices of the max values before every downsampling in the encoder part and to do an upsampling with the help of those "masks" of indices to restore only the values in their most relevant positions.

Edit the main.py file only, add the path to the image you want to study, number of layer and feature map you want to visualize. 

There is an example image in the folder example_data. 
 
