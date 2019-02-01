#!/usr/bin/env python
# coding: utf-8


# Imports

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, GRU, LSTM
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical, plot_model
import collections



# Set random seed for consistency in testing

randomseed=4
from numpy.random import seed
seed(randomseed)



# Splits data in train and test sets

def split_train_test(X_total,Y_total,frac_train=0.9):
    number_examples = int(X_total.shape[0])
    X_train = X_total[:int(number_examples*frac_train)]
    X_test = X_total[int(number_examples*frac_train):]
    Y_train = Y_total[:int(number_examples*frac_train)]
    Y_test = Y_total[int(number_examples*frac_train):]
    return X_train,X_test,Y_train,Y_test


# Prepares the data

def preparator(arr_orig):

    # Transforms the ghost vars into numpy arrays s.t. can be used by keras

    number_examples = len(arr_orig)
    new_arr = []
    
    for i in range(number_examples):
        new_arr.append(arr_orig[i])
    
    new_arr = np.array(new_arr)
    return new_arr



# Import data

all_tracks = pd.read_pickle('../all_hits.pkl')
all_tracks = all_tracks.sample(frac=1,random_state=randomseed).reset_index(drop=True)     # Shuffles the data


# Prepare data:     Target is 0 or 1 (ghost or real)
#                   Variables is a list of 22 parameters

target = all_tracks['mcpid']
variables = all_tracks['ghost_vars']


X_total = np.array(variables)
Y_total = np.array(target)


X_total = preparator(X_total)

# print(X_total.shape)                              # Is (118545, 22)
(num_examples, laenge) = X_total.shape

X_train,X_test,Y_train,Y_test = split_train_test(X_total,Y_total)     # Split into train and test set


# Define presision and recall to be used as metric in training

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


# Define inverse square root unit, to be used as activation function

def isru(x):
	return x/keras.backend.sqrt(1+keras.backend.square(x))


# Define the model

model = Sequential()
model.add(Dense(27, input_dim=laenge, activation=isru))
model.add(Dense(27, activation=isru))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # metrics = [precision, recall]

print(model.summary())

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=100, verbose=0) # verbose=1: progress bar, verbose=0: silent


# Plot the model graph

#plot_model(model, to_file='mlp_graph.png')


scores = model.evaluate(X_test, Y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1]*100))


# Save accuracy into txt file if necessary

#with open("accuracy.txt","a+") as f:
#   f.write('Accuracy: %.2f%% \n' % (scores[1]*100))


# Make predicitons with trained model

Y_pred = model.predict(X_test).ravel()


# Tools to plot ROC curve

fpr, tpr, thresholds = roc_curve(Y_test.tolist(), Y_pred.tolist())    	# False positive rate and true positive rate for each possible threshold
areaundercurve = auc(fpr, tpr)											# Also calculate area under curve


# Plot actual ROC curve

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(areaundercurve))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
# plt.savefig('ROC_MLP',dpi=300)
plt.show()





