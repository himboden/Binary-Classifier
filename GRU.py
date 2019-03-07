#!/usr/bin/env python
# coding: utf-8


# Imports

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.utils import shuffle
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, GRU, LSTM
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical, plot_model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import collections



# Define random seed for consistency in testing

random_seed = 42
np.random.seed(random_seed)



# Helper function

def preparator(X_input):
    
    # Takes the dataframe and makes it (total_examples, max_len_example, 3)-dimensional
        
    num_examples = int(len(X_input))
    maxlen = 50
    stacked = []

    desired_shape = (maxlen,3)
    for i in range(num_examples):      # This stacks and pads the data to shape 3x50
        tempo = np.stack((X_input['hits_x'].values[i], X_input['hits_y'].values[i],
                          X_input['hits_z'].values[i]), axis=-1)
        nullen = np.zeros(desired_shape)
        nullen[:tempo.shape[0],:3]=tempo
        stacked.append(nullen)
    
    stacked = np.array(stacked)
    return stacked
                             


# Load data

all_tracks = pd.read_pickle('../upgrade_hits.pkl')  



# Sort and prepare data

Y_total = all_tracks['mcpid'].values   # Is 1 if real and 0 if ghost

X_total = preparator(all_tracks)


# Split train and test set

X_train, X_test, Y_train, Y_test = train_test_split(X_total, Y_total, test_size=0.2, random_state=random_seed)



# Define precision and recall, to be used as a metric later

import tensorflow as tf

def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

precision = as_keras_metric(tf.metrics.precision)
recall = as_keras_metric(tf.metrics.recall)



# Define the model

model = Sequential()
model.add(Bidirectional(GRU(50,return_sequences=True, activation = 'selu'),input_shape=(50,3)))  
model.add(BatchNormalization())   
model.add(Dropout(0.1))
model.add(Bidirectional(GRU(50, activation = 'selu')))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[precision, recall])

print(model.summary())
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=75, verbose = 1)  # callbacks=[tensorboard] 

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# Plot a model of the network

#plot_model(model, to_file='GRU_model.png')

# Plot the loss if desired

#plt.figure(1)
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Model losses')
#plt.xlabel('epoch')
#plt.ylabel('loss')
#plt.legend(['train','test'], loc='upper right')
#plt.savefig('loss.png',dpi=300)


# Make prediciton
    
Y_pred = model.predict(X_test).ravel()        


# ROC curve 

fpr, tpr, thresholds = roc_curve(Y_test.tolist(), Y_pred.tolist())    	# False positive rate and true positive rate for each possible threshold
areaundercurve = auc(fpr, tpr)											# Also calculate area under curve


plt.figure(2)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(areaundercurve))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
# plt.savefig('ROC_GRU.png')
plt.show()



