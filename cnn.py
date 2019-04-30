# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 18:52:33 2019

@author: Soumya Kumari
"""
import os, cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras import backend as K
# K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, RMSprop, Adam

# %%

PATH = os.getcwd()
# Define data path
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

img_rows = 128
img_cols = 128
num_channel = 3
num_epoch = 20

# Define the number of classes
num_classes = 20

labels_name = {'Apple': 0, 'Banana': 1, 'Bean': 2, 'Bread': 3, 'Burger': 4, 'Cup_cakes': 5, 'Donuts': 6, 'Dumpling': 7, 'Egg': 8,
               'French_fries': 9, 'Garlic_bread': 10, 'Orange': 11, 'Pancakes':12, 'Pasta': 13, 'Pizza': 14, 'Samosa': 15, 'Spring_rolls':16, 'Tomato':17, 'Waffles': 18, 'Watermelon':19}
type(labels_name)

img_data_list = []
labels_list = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loading the images of dataset-' + '{}\n'.format(dataset))
    label = labels_name[dataset]
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (128, 128))
        img_data_list.append(input_img_resize)
        labels_list.append(label)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print(img_data.shape)

labels = np.array(labels_list)
# print the count of number of samples for different classes
print(np.unique(labels, return_counts=True))
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

# Shuffle the dataset
x, y = shuffle(img_data, Y, random_state=2)
# Split the dataset
# x= np.expand_dims(x, axis = 0)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
X_train = X_train.reshape(6422, 128, 128, 3)
X_test = X_test.reshape(1606, 128, 128, 3)

if num_channel == 1:
    if K.image_dim_ordering() == 'th':
        img_data = np.expand_dims(img_data, axis=1)
        print(img_data.shape)
    else:
        img_data = np.expand_dims(img_data, axis=4)
        print(img_data.shape)

else:
    if K.image_dim_ordering() == 'th':
        img_data = np.rollaxis(img_data, 3, 1)
        print(img_data.shape)

# %%
# Defining the model
input_shape = img_data[0].shape
print(input_shape)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Dropout(0.5))


model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Dropout(0.5))


'''
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(LeakyReLU(alpha = 0.1))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))


model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
'''


model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(LeakyReLU(alpha = 0.1))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

# Viewing model_configuration

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

# %%
# Training
hist = model.fit(X_train, y_train, batch_size=60, nb_epoch=150, verbose=1, validation_data=(X_test, y_test))

# hist = model.fit(X_train, y_train, batch_size=32, nb_epoch=20,verbose=1, validation_split=0.2)



# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(len(train_acc))

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])


# %%

# Evaluating the model

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

'''
test_image = X_test[0:1]
print (test_image.shape)

print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])
'''

# Testing a new image
test_image = cv2.imread('C:/Users/ADMIN/Documents/pasta1.jpg')

refPt = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(test_image, refPt[0], refPt[1], (0, 255, 0), 2)
    # cv2.imshow("image", image)


# load the image, clone it, and setup the mouse callback function


clone = test_image.copy()
cv2.namedWindow("test_image")
cv2.setMouseCallback("test_image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("test_image", test_image)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        test_image = clone.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    cv2.imshow("ROI", roi)
    cv2.imwrite("G:/test.jpg", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image = cv2.resize(test_image, (128, 128))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print(test_image.shape)

if num_channel == 1:
    if K.image_dim_ordering() == 'th':
        test_image = np.expand_dims(test_image, axis=0)
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)
    else:
        test_image = np.expand_dims(test_image, axis=3)
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)

else:
    if K.image_dim_ordering() == 'th':
        test_image = np.rollaxis(test_image, 2, 0)
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)
    else:
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)

# Predicting the test image
print((model.predict(test_image)))
print(model.predict_classes(test_image))


# %%

# Visualizing the intermediate layer

#
def get_featuremaps(model, layer_idx, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output, ])
    activations = get_activations([X_batch, 0])
    return activations


layer_num = 3
filter_num = 0

activations = get_featuremaps(model, int(layer_num), test_image)

print(np.shape(activations))
feature_maps = activations[0][0]
print(np.shape(feature_maps))

if K.image_dim_ordering() == 'th':
    feature_maps = np.rollaxis((np.rollaxis(feature_maps, 2, 0)), 2, 0)
print(feature_maps.shape)

fig = plt.figure(figsize=(16, 16))
plt.imshow(feature_maps[:, :, filter_num])
plt.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num) + '.jpg')

num_of_featuremaps = feature_maps.shape[2]
fig = plt.figure(figsize=(16, 16))
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num = int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
    ax = fig.add_subplot(subplot_num, subplot_num, i + 1)
    # ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
    ax.imshow(feature_maps[:, :, i])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
plt.show()
fig.savefig("featuremaps-layer-{}".format(layer_num) + '.png')

# %%
# Printing the confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
# y_pred = model.predict_classes(X_test)
# print(y_pred)
target_names = ['Class 0(Apple)', 'Class 1(Banana)', 'Class 2(Bean)', 'Class 3(Bread)', 'Class 4(Burger)', 'Class 5(Cup_cakes)', 'Class 6(Donuts)', 'Class 7(Dumpling)', 'Class 8(Egg)',
               'Class 9(French_fries)', 'Class 10(Garlic_bread)', 'Class 11(Orange)', 'Class 12(Pancakes)', 'Class 13(Pasta', 'Class 14(Pizza)', 'Class 15(Samosa)', 'Class 16(Spring_rolls)', 'Class 17(Tomato)', 'Class 18(Waffles)', 'Class 19(Watermelon)']

print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))

print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))


# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test, axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
# plt.figure()
# Plot normalized confusion matrix
# plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
#                      title='Normalized confusion matrix')
# plt.figure()
plt.show()

# %%
# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

model.save('model_exp33.hdf5')
loaded_model = load_model('model_exp32.hdf5')