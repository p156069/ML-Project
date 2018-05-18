import numpy as np
import os
import cv2
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential, save_model
from keras.utils import np_utils
from sklearn.utils import shuffle


path = './dataset/'#Set path where images are saved

gestures = os.listdir(path)[1:] #Get path of saved images

x_ , y_ = [], []


#From each image path load images into Numpy array and assign label
for i in gestures:
    images = os.listdir(path + i)
    for j in images:
        img_path = path + i + '/' + j
        img = cv2.imread(img_path, 0)
        img = np.array(img)
        img = img.reshape( (50,50,1) )
        img = img/255.0
        x_.append(img)
        y_.append( int(i) )


x = np.array(x_)
y = np.array(y_)

#Converts a class vector (integers) to binary class matrix.
y = np_utils.to_categorical(y)
no_classes = y.shape[1]

#Shuffle dataset to make it more generalize
x , y = shuffle(x, y, random_state=0)

#Splitting training and testing dataset
split = int( 0.6*( x.shape[0] ) )
t_feautres = x[ :split ]
t_labels = y[ :split ]
t2_feautres = x[ split: ]
t2_labels = y[ split: ]


#Creating Sequenrial model
model = Sequential()

#Adding and applying 32 convolution filters 3x3 dimension
model.add( Convolution2D(32, 3, 3, input_shape = (50,50,1) ) )
#Applying Relu activiaction function
model.add( Activation('relu') )
#Adding and apply 64 convolution filters of 3x3 dimension in 2nd hidden layer
model.add( Convolution2D( 64,3,3 ) )
#Applying Relu activiaction function
model.add( Activation('relu') )
#Doing Maxpooling of window size 2x2 in 3rd hidden layer
model.add( MaxPooling2D( pool_size=(2,2) ) )
#Adding and applying 16 convolution filters 3x3 dimension in 4th layer
model.add( Convolution2D( 16, 3, 3 ) )
#Applying Relu activiaction function
model.add( Activation('relu') )
#Flattening the perceptrons in next layer
model.add( Flatten() )
#Setting dropout rate to 0.25 for good generalization
model.add( Dropout(0.25) )
model.add( Dense(no_classes) )
#Applying softmax at last or output layer
model.add( Activation('softmax') )
#Printing summery of model
model.summary()
#Compiling model using Adam optimizer and for loss using categorical crossentropy and using metric of accuracy
model.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )
#Fitting model on dataset with batch size 128 and with 3 epoch
model.fit( t_feautres, t_labels, validation_data=( t2_feautres, t2_labels ), shuffle=True, batch_size=128, nb_epoch=3 )
#Saving the model
model.save('model.h5')

