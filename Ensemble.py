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
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Bidirectional, GRU
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.utils import to_categorical, plot_model
import collections





# Define random seed for consisency in testing

random_seed = 42
np.random.seed(random_seed)



# Helper function

def preparator(X_input, the_dimension):
    
    # This is for the RNN-part of the network
    # Takes the dataframe and makes it (total_examples, max_len_example, features_per_timestep)-dimensional
    # The last dimension can be the following:
        # 3: Take x,y,z positions in each step
        # 4: Take x,y,z positions and momentum in each step
        
    num_examples = int(len(X_input))
    maxlen = 50
    stacked = []
    
    if the_dimension == 3:
        desired_shape = (maxlen,the_dimension)
        for i in range(num_examples):      # This stacks and pads the data to shape 3x50
            tempo = np.stack((X_input['hits_x'].values[i], X_input['hits_y'].values[i],
                              X_input['hits_z'].values[i]), axis=-1)
            nullen = np.zeros(desired_shape)
            nullen[:tempo.shape[0],:the_dimension]=tempo
            stacked.append(nullen)
                             
    if the_dimension == 4:
        desired_shape = (maxlen,the_dimension)
        for i in range(num_examples):      # This stacks and pads the data to shape 4x50
            tempo = np.stack((X_input['hits_x'].values[i], X_input['hits_y'].values[i],
                              X_input['hits_z'].values[i], X_input['MCmom'].values[i]), axis=-1)
            nullen = np.zeros(desired_shape)
            nullen[:tempo.shape[0],:the_dimension]=tempo
            stacked.append(nullen)
    
    stacked = np.array(stacked)
    return stacked  




def array_preparator(arr_orig):
    
    # This is for the Feedforward-part of the network
    # Transforms the ghost vars into numpy arrays s.t. can be used by keras
    
    number_examples = len(arr_orig)
    new_arr = []
    
    for i in range(number_examples):
        new_arr.append(arr_orig[i])
    
    new_arr = np.array(new_arr)
    return new_arr





# Load data

all_tracks = pd.read_pickle('../upgrade_hits.pkl')   



# Sort and prepare data

Y_total = all_tracks['mcpid'].values

X_total_GRU = preparator(all_tracks,4)    # 3 if x,y,z only
X_total_MLP = array_preparator(np.array(all_tracks['ghost_vars'].values))

laenge = len(X_total_MLP[0])

X_train_GRU, X_test_GRU, Y_train_GRU, Y_test_GRU = train_test_split(X_total_GRU, Y_total, test_size=0.2, random_state=random_seed)
X_train_MLP, X_test_MLP, Y_train_MLP, Y_test_MLP = train_test_split(X_total_MLP, Y_total, test_size=0.2, random_state=random_seed)



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


# # Define the total model (GRU and MLP merged)

# Defines the input of the FF-part
mlp_in = Input(shape=(laenge,),name='mlp_in')

# Defines the model for the RNN-part
gru_in = Input(shape=(50,4),name='gru_in')   # (50,3) if x,y,z only
bidir_1 = Bidirectional(GRU(30, return_sequences=True, activation='selu'))(gru_in)
norm_1 = BatchNormalization()(bidir_1)
drop_1 = Dropout(0.1)(norm_1)
bidir_2 = Bidirectional(GRU(30, activation='selu'))(drop_1)
norm_2 = BatchNormalization()(bidir_2)
drop_2 = Dropout(0.1)(norm_2)
gru_out = Dense(1)(drop_2)

# Merge output of RRN with input of FF
merged = concatenate([mlp_in, gru_out])

# Defines the model for the FF-part
dense_1 = Dense(27, activation='relu')(merged)
dense_2 = Dense(27, activation='relu')(dense_1)
out = Dense(1, activation='sigmoid')(dense_2)

model = Model(inputs=[mlp_in, gru_in], outputs=out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[precision, recall])

print(model.summary())
history = model.fit([X_train_MLP, X_train_GRU], Y_train_GRU, validation_data=([X_test_MLP, X_test_GRU], Y_test_GRU), epochs=3, batch_size=75, verbose = 1)


scores = model.evaluate([X_test_MLP,X_test_GRU], Y_test_GRU, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# Plot a model of the network

plot_model(model, to_file='Ensemble_model.png')


# Make prediction
    
Y_pred = model.predict([X_test_MLP, X_test_GRU]).ravel()


# ROC curve

fpr, tpr, thresholds = roc_curve(Y_test_GRU.tolist(), Y_pred.tolist())     # False positive rate and true positive rate for each possible threshold
areaundercurve = auc(fpr, tpr)											    # Also calculate area under curve


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(areaundercurve))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('ROC_unbalanced')
plt.show()





