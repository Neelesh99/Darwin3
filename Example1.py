import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.optimizers import rmsprop
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras import regularizers
import Darwin3 as D3


# the data, shuffled and split between train and val sets
# Here we are using the official test set as our validation set, in further
# tutorials, test and validation splits will be explained properly.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('Image shape: {0}'.format(x_train.shape[1:]))
print('Total number of training samples: {0}'.format(x_train.shape[0]))
print('Total number of test samples: {0}'.format(x_test.shape[0]))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize the image
x_train /= 255
x_test /= 255

y_train_class = np_utils.to_categorical(y_train, 10)
y_test_class = np_utils.to_categorical(y_test, 10)

# Define here your architecture

#Defining Initial Population for Genetic Algorithm

# CONVOLUTIONAL_ACTIVATION: Filter Count, Convolutional Dimension, Padding Type, Convolutional Stride, Convolutional Activation
# POOLING: Pooling Type, Pooling Dimension, Pooling Strides
# DENSE_ACTIVATION: Node Count, Activation type
# BATCH_NORMALISATION: _
# DROPOUT: Rate as (0->100)

S = [
     ([[D3.CONVOLUTIONAL_ACTIVATION, 32, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 32, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 20],
       [D3.CONVOLUTIONAL_ACTIVATION, 64, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 64, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 20],
       [D3.CONVOLUTIONAL_ACTIVATION, 128, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 128, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 20]],
       [[D3.DENSE_ACTIVATION, 12, 0]]),
     #===========================================#
      ([[D3.CONVOLUTIONAL_ACTIVATION, 32, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 32, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 10],
       [D3.CONVOLUTIONAL_ACTIVATION, 78, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 78, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 50],
       [D3.CONVOLUTIONAL_ACTIVATION, 128, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 128, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 20]],
       []),
     #============================================#
      ([[D3.CONVOLUTIONAL_ACTIVATION, 64, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 64, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 10],
       [D3.CONVOLUTIONAL_ACTIVATION, 128, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 128, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 50],
       [D3.CONVOLUTIONAL_ACTIVATION, 256, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 256, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 20]],
       [[D3.DENSE_ACTIVATION, 10, 0]]),
     #============================================#
      ([[D3.CONVOLUTIONAL_ACTIVATION, 64, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 64, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 10],
       [D3.CONVOLUTIONAL_ACTIVATION, 128, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 128, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 50],
       [D3.CONVOLUTIONAL_ACTIVATION, 256, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 256, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 20]],
       [[D3.DENSE_ACTIVATION, 10, 0]]),
     #============================================#
      ([[D3.CONVOLUTIONAL_ACTIVATION, 16, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 16, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 10],
       [D3.CONVOLUTIONAL_ACTIVATION, 32, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 32, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 30],
       [D3.CONVOLUTIONAL_ACTIVATION, 64, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 64, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 60],
       [D3.CONVOLUTIONAL_ACTIVATION, 128, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 128, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 40],
       [D3.CONVOLUTIONAL_ACTIVATION, 256, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 256, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 20]],
       [[D3.DENSE_ACTIVATION, 10, 0]]),
     #============================================#
      ([[D3.CONVOLUTIONAL_ACTIVATION, 64, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 64, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 10],
       [D3.CONVOLUTIONAL_ACTIVATION, 128, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 50],
       [D3.CONVOLUTIONAL_ACTIVATION, 256, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 256, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.CONVOLUTIONAL_ACTIVATION, 256, 2, 1, 1, 2],
       [D3.BATCH_NORMALISATION],
       [D3.POOLING, 0, 2, 1],
       [D3.DROPOUT, 20]],
       [[D3.DENSE_ACTIVATION, 10, 0]]),
     #============================================#
     ([[D3.CONVOLUTIONAL_ACTIVATION, 64, 2, 1, 2, 2], [D3.POOLING, 0, 3, 1], [D3.CONVOLUTIONAL_ACTIVATION, 128, 2, 1, 2, 2], [D3.POOLING, 0, 2, 1]],[[D3.DENSE_ACTIVATION, 10, 0], [D3.DENSE_ACTIVATION, 12, 1]])]


Gene_Pool = D3.Genetic_Engine(S,x_train,y_train_class, x_test, y_test_class, D3.CLASSIFICATION)
model = Gene_Pool.run(5)

model.summary()

# . . .


# initiate RMSprop optimizer
opt = rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(x_train, y_train_class, batch_size=32, epochs=20)

score = model.evaluate(x_test, y_test_class, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])