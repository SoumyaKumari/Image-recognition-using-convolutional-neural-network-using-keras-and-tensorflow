# Image recognition using convolutional neural network using keras
This is a code for recognizing 20 categories of food images and gives the accuracy and loss graphs with confusion matrix and one can  visualize the intermediate layers as well. 
Firstly, dataset is loaded (dataset should be placed in the ame directory as your project) and images are preprocessed. CNN model is complied and thus training phase starts. For testing on a particular image, a bounding-box concept has been used where user can crop the image (press c to confirm the box) to remove backgroung noise.
Currently, this codde gives the validation accuracy of 76.5% and loss of 1.089 at 160th epoch. 
No. of layers - 6
