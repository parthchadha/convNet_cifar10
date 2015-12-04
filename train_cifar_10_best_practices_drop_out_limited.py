#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.
This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html
More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function
import theano.sandbox.cuda

import sys
import os
import time
import sklearn.cross_validation
import sklearn.metrics

import numpy as np
import theano
import theano.tensor as T
from load_cifar import load_cifar10
import cPickle as pickle

import lasagne
# gpu_number = sys.argv[3]
# gpu_use = "gpu" + str(gpu_number);
# theano.sandbox.cuda.use(gpu_use)

#theano.config.floatX='float32'
EPOCHS  = 10
BATCH_SIZE = 100
save_path = '/output/'
DIM = 32
CATEGORIES = 10

def build_model_4_2(input_var=None):
	#
	#
	# 4_1 - >  weights glorotUniform
	# 4_2 - > weights heuniform

	# Input layer, as usual:
	l_in = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
		input_var=input_var)
	# This time we do not apply input dropout, as it tends to work less well
	# for convolutional layers.

	# l_in_pad = lasagne.layers.PadLayer(
	#  	l_in,
	#  	width=2,#padding width
	#  )

	#drop_0 = lasagne.layers.DropoutLayer(l_in, p=0.2)

	# Convolutional layer with 32 kernels of size 5x5. Strided and padded
	# convolutions are supported as well; see the docstring.
	conv_1 = lasagne.layers.Conv2DLayer(
			l_in, num_filters=32, filter_size=(8, 8),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu')			
			)
	# Expert note: Lasagne provides alternative convolutional layers that
	# override Theano's choice of which implementation to use; for details
	# please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.
	print ("shape after conv_1",(lasagne.layers.get_output_shape(conv_1) ) )

	# Max-pooling layer of factor 2 in both dimensions:
#	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
	pool_1 = lasagne.layers.MaxPool2DLayer(conv_1, pool_size=(2, 2),stride = 1)
	print ("shape after pool_1 ",(lasagne.layers.get_output_shape(pool_1) ) )

	# pool_1_pad = lasagne.layers.PadLayer(
	#  	pool_1,
	#  	width=2,#padding width
	#  )
	#
	#drop_1 = lasagne.layers.DropoutLayer(pool_1, p=0.5)

	# Another convolution with 32 5x5 kernels, and another 2x2 pooling:
	conv_2 = lasagne.layers.Conv2DLayer(
			pool_1, num_filters=48, filter_size=(5, 5),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu')
			)
	#network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
	pool_2 = lasagne.layers.Pool2DLayer(conv_2, pool_size=(2, 2),stride = 1)
	print ("shape after pool_2",(lasagne.layers.get_output_shape(pool_2) ) )

	# pool_2_pad = lasagne.layers.PadLayer(
	#  	pool_2,
	#  	width=1,#padding width
	#  )

	#drop_2 = lasagne.layers.DropoutLayer(pool_2, p=0.5)

	# Another convolution with 32 5x5 kernels, and another 2x2 pooling:
	conv_3 = lasagne.layers.Conv2DLayer(
			pool_2, num_filters=64, filter_size=(3, 3),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu')
			)
	
	print ("shape after conv_3 ",(lasagne.layers.get_output_shape(conv_3) ) )
	
	#drop_3 = lasagne.layers.DropoutLayer(conv_3, p=0.5)

	conv_4 = lasagne.layers.Conv2DLayer(
			conv_3, num_filters=64, filter_size=(3, 3),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu')
			)
	
	#drop_4 = lasagne.layers.DropoutLayer(conv_4, p=0.5)

	print ("shape after conv_4 ",(lasagne.layers.get_output_shape(conv_4) ) )
	conv_5 = lasagne.layers.Conv2DLayer(
			conv_4, num_filters=48, filter_size=(3, 3),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu')
			)
	
	print ("shape after conv_5 ",(lasagne.layers.get_output_shape(conv_5) ) )


	pool_3 = lasagne.layers.Pool2DLayer(conv_5, pool_size=(2, 2),stride = 1)
	print ("shape after pool_3",(lasagne.layers.get_output_shape(pool_3) ) )
	#drop_5 = lasagne.layers.DropoutLayer(pool_3, p=0.5)

	# And, finally, the 10-unit output layer with 50% dropout on its inputs:
	hidden_1 = lasagne.layers.DenseLayer(
	 		#lasagne.layers.dropout(network, p=.5),
	 		pool_3,
	 		num_units=500,
	 		nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu')
	 		)
	drop_6 = lasagne.layers.DropoutLayer(hidden_1, p=0.5)

	print ("shape after hidden_1 ",(lasagne.layers.get_output_shape(hidden_1) ) )
	hidden_2 = lasagne.layers.DenseLayer(
	 		#lasagne.layers.dropout(network, p=.5),
	 		drop_6,
	 		num_units=500,
	 		nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu')
	 		)
	drop_7 = lasagne.layers.DropoutLayer(hidden_2, p=0.5)
	
	print ("shape after hidden_2 ",(lasagne.layers.get_output_shape(hidden_2) ) )

	network = lasagne.layers.DenseLayer(
	 		#lasagne.layers.dropout(network, p=.5),
	 		drop_7,
	 		num_units=10,
	 		nonlinearity=lasagne.nonlinearities.softmax,
	 		W=lasagne.init.GlorotUniform('relu')
	 		)

	return network

def build_model_3_2(input_var=None):
	#
	#
	#
	# 

	# Input layer, as usual:
	l_in = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
		input_var=input_var)
	# This time we do not apply input dropout, as it tends to work less well
	# for convolutional layers.

	# l_in_pad = lasagne.layers.PadLayer(
	#  	l_in,
	#  	width=2,#padding width
	#  )

	#drop_0 = lasagne.layers.DropoutLayer(l_in, p=0.2)

	# Convolutional layer with 32 kernels of size 5x5. Strided and padded
	# convolutions are supported as well; see the docstring.
	conv_1 = lasagne.layers.Conv2DLayer(
			l_in, num_filters=64, filter_size=(5, 5),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu'))
	# Expert note: Lasagne provides alternative convolutional layers that
	# override Theano's choice of which implementation to use; for details
	# please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.
	print ("shape after conv_1",(lasagne.layers.get_output_shape(conv_1) ) )

	# Max-pooling layer of factor 2 in both dimensions:
#	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
	pool_1 = lasagne.layers.MaxPool2DLayer(conv_1, pool_size=(2, 2),stride = 1)
	print ("shape after pool_1 ",(lasagne.layers.get_output_shape(pool_1) ) )

	# pool_1_pad = lasagne.layers.PadLayer(
	#  	pool_1,
	#  	width=2,#padding width
	#  )
	#drop_1 = lasagne.layers.DropoutLayer(pool_1, p=0.5)

	# Another convolution with 32 5x5 kernels, and another 2x2 pooling:
	conv_2 = lasagne.layers.Conv2DLayer(
			pool_1, num_filters=64, filter_size=(5, 5),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu')
			)
	#network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
	pool_2 = lasagne.layers.Pool2DLayer(conv_2, pool_size=(3, 3),stride = 2)
	print ("shape after pool_2",(lasagne.layers.get_output_shape(pool_2) ) )

	# pool_2_pad = lasagne.layers.PadLayer(
	#  	pool_2,
	#  	width=1,#padding width
	#  )

	#drop_2 = lasagne.layers.DropoutLayer(pool_2, p=0.5)

	# Another convolution with 32 5x5 kernels, and another 2x2 pooling:
	conv_3 = lasagne.layers.Conv2DLayer(
			pool_2, num_filters=128, filter_size=(5, 5),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu')
			)
	
	print ("shape after conv_3 ",(lasagne.layers.get_output_shape(conv_3) ) )

	pool_3 = lasagne.layers.Pool2DLayer(conv_3, pool_size=(3, 3),stride = 2)
	print ("shape after pool_3",(lasagne.layers.get_output_shape(pool_3) ) )
	#drop_3 = lasagne.layers.DropoutLayer(pool_3, p=0.5)

	# And, finally, the 10-unit output layer with 50% dropout on its inputs:
	hidden_1 = lasagne.layers.DenseLayer(
	 		#lasagne.layers.dropout(network, p=.5),
	 		pool_3,
	 		num_units=3072,
	 		nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu')
	 		)
	drop_4 = lasagne.layers.DropoutLayer(hidden_1, p=0.5)
	
	print ("shape after hidden_1 ",(lasagne.layers.get_output_shape(hidden_1) ) )
	hidden_2 = lasagne.layers.DenseLayer(
	 		#lasagne.layers.dropout(network, p=.5),
	 		drop_4,
	 		num_units=2048,
	 		nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu')
	 		)
	
	print ("shape after hidden_2 ",(lasagne.layers.get_output_shape(hidden_2) ) )
	drop_5 = lasagne.layers.DropoutLayer(hidden_2, p=0.5)

	network = lasagne.layers.DenseLayer(
	 		#lasagne.layers.dropout(network, p=.5),
	 		drop_5,
	 		num_units=10,
	 		nonlinearity=lasagne.nonlinearities.softmax,
	 		W=lasagne.init.GlorotUniform('relu')
			)


	return network

def build_model_2_2(input_var=None):
	#
	#
	#
	# 

	# Input layer, as usual:
	l_in = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
		input_var=input_var)
	# This time we do not apply input dropout, as it tends to work less well
	# for convolutional layers.

	# l_in_pad = lasagne.layers.PadLayer(
	#  	l_in,
	#  	width=2,#padding width
	#  )
	#drop_0 = lasagne.layers.DropoutLayer(l_in, p=0.2)


	# Convolutional layer with 32 kernels of size 5x5. Strided and padded
	# convolutions are supported as well; see the docstring.
	conv_1 = lasagne.layers.Conv2DLayer(
			l_in, num_filters=96, filter_size=(5, 5),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu'))
	# Expert note: Lasagne provides alternative convolutional layers that
	# override Theano's choice of which implementation to use; for details
	# please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.
	print ("shape after conv_1",(lasagne.layers.get_output_shape(conv_1) ) )

	# Max-pooling layer of factor 2 in both dimensions:
#	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
	pool_1 = lasagne.layers.MaxPool2DLayer(conv_1, pool_size=(3, 3),stride = 2)
	print ("shape after pool_1 ",(lasagne.layers.get_output_shape(pool_1) ) )

	# pool_1_pad = lasagne.layers.PadLayer(
	#  	pool_1,
	#  	width=2,#padding width
	#  )
	#drop_1 = lasagne.layers.DropoutLayer(pool_1, p=0.8)

	# Another convolution with 32 5x5 kernels, and another 2x2 pooling:
	conv_2 = lasagne.layers.Conv2DLayer(
			pool_1, num_filters=192, filter_size=(5, 5),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu')
			)
	#network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
	pool_2 = lasagne.layers.Pool2DLayer(conv_2, pool_size=(3, 3),stride = 2)
	print ("shape after pool_2",(lasagne.layers.get_output_shape(pool_2) ) )

	# pool_2_pad = lasagne.layers.PadLayer(
	#  	pool_2,
	#  	width=1,#padding width
	#  )

	#drop_2 = lasagne.layers.DropoutLayer(pool_2, p=0.5)

	# Another convolution with 32 5x5 kernels, and another 2x2 pooling:
	conv_3 = lasagne.layers.Conv2DLayer(
			pool_2, num_filters=160, filter_size=(3, 3),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu')
			)
	
	print ("shape after conv_3 ",(lasagne.layers.get_output_shape(conv_3) ) )

	pool_3 = lasagne.layers.Pool2DLayer(conv_3, pool_size=(2, 2),stride = 2)
	print ("shape after pool_3",(lasagne.layers.get_output_shape(pool_3) ) )
	#drop_3 = lasagne.layers.DropoutLayer(pool_3, p=0.5)

	# And, finally, the 10-unit output layer with 50% dropout on its inputs:
	hidden_1 = lasagne.layers.DenseLayer(
	 		#lasagne.layers.dropout(network, p=.5),
	 		pool_3,
	 		num_units=500,
	 		nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu')
	 		)
	
	print ("shape after hidden_1 ",(lasagne.layers.get_output_shape(hidden_1) ) )
	drop_4 = lasagne.layers.DropoutLayer(hidden_1, p=0.5)

	network = lasagne.layers.DenseLayer(
	 		#lasagne.layers.dropout(network, p=.5),
	 		drop_4,
	 		num_units=10,
	 		nonlinearity=lasagne.nonlinearities.softmax,
	 		W=lasagne.init.GlorotUniform('relu')
	 		)



	return network

def build_model_1_2(input_var=None):
	#
	#
	#
	# 

	# Input layer, as usual:
	l_in = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
		input_var=input_var)
	# This time we do not apply input dropout, as it tends to work less well
	# for convolutional layers.

	# l_in_pad = lasagne.layers.PadLayer(
	#  	l_in,
	#  	width=2,#padding width
	#  )
	#drop_0 = lasagne.layers.DropoutLayer(l_in, p=0.2)


	# Convolutional layer with 32 kernels of size 5x5. Strided and padded
	# convolutions are supported as well; see the docstring.
	conv_1 = lasagne.layers.Conv2DLayer(
			l_in, num_filters=64, filter_size=(5, 5),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu'))
	# Expert note: Lasagne provides alternative convolutional layers that
	# override Theano's choice of which implementation to use; for details
	# please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.
	print ("shape after conv_1",(lasagne.layers.get_output_shape(conv_1) ) )

	# Max-pooling layer of factor 2 in both dimensions:
#	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
	pool_1 = lasagne.layers.MaxPool2DLayer(conv_1, pool_size=(2, 2),stride = 2)
	print ("shape after pool_1 ",(lasagne.layers.get_output_shape(pool_1) ) )

	# pool_1_pad = lasagne.layers.PadLayer(
	#  	pool_1,
	#  	width=2,#padding width
	#  )
	#drop_1 = lasagne.layers.DropoutLayer(pool_1, p=0.8)

	# Another convolution with 32 5x5 kernels, and another 2x2 pooling:
	conv_2 = lasagne.layers.Conv2DLayer(
			pool_1, num_filters=96, filter_size=(5, 5),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu')
			)
	#network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
	pool_2 = lasagne.layers.Pool2DLayer(conv_2, pool_size=(2, 2),stride = 2)
	print ("shape after pool_2",(lasagne.layers.get_output_shape(pool_2) ) )

	# pool_2_pad = lasagne.layers.PadLayer(
	#  	pool_2,
	#  	width=1,#padding width
	#  )

	#drop_2 = lasagne.layers.DropoutLayer(pool_2, p=0.5)

	# Another convolution with 32 5x5 kernels, and another 2x2 pooling:
	conv_3 = lasagne.layers.Conv2DLayer(
			pool_2, num_filters=160, filter_size=(5, 5),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu')
			)
	
	print ("shape after conv_3 ",(lasagne.layers.get_output_shape(conv_3) ) )

	#drop_3 = lasagne.layers.DropoutLayer(conv_3, p=0.5)

	# And, finally, the 10-unit output layer with 50% dropout on its inputs:
	hidden_1 = lasagne.layers.DenseLayer(
	 		#lasagne.layers.dropout(network, p=.5),
	 		conv_3,
	 		num_units=1000,
	 		nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform('relu')
	 		)
	
	print ("shape after hidden_1 ",(lasagne.layers.get_output_shape(hidden_1) ) )
	drop_4 = lasagne.layers.DropoutLayer(hidden_1, p=0.5)

	network = lasagne.layers.DenseLayer(
	 		#lasagne.layers.dropout(network, p=.5),
	 		drop_4,
	 		num_units=10,
	 		nonlinearity=lasagne.nonlinearities.softmax,
	 		W=lasagne.init.GlorotUniform('relu')
	 		)


	return network

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
	
def main(x = 1, save=1, num_epochs=100,alpha=0.001,mv=0.9,l2_rate = 0.001):
	# Load the dataset
	print("Loading data...")
	learning_rate_decay = 0.1
	epoch_delay = [100,150,200,500]
	#epoch_delay = [2,3,4]
	
	epoch_delay_count = 0
	#X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
	X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
	
	#l2_regularization_rate = 0.001
	l2_regularization_rate = l2_rate
	# Prepare Theano variables for inputs and targets
	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')

	# Create neural network model (depending on first command line parameter)
	print("Building model and compiling functions...")
	
	if x == 1:
		network = build_model_1_2(input_var)
		print("model 1")
    # Do the thing
	elif x == 2:
		network = build_model_2_2(input_var)
		print("model 2")
    # Do the other thing
	elif x == 3:
		network = build_model_3_2(input_var)
		print("model 3")
    # Fall-through by not using elif, but now the default case includes case 'a'!
	elif x == 4:
		network = build_model_4_2(input_var)
		print("model 4")

	

	#learning_rate_alpha = theano.shared(np.float32(alpha))
	learning_rate_alpha = alpha
	stochastic_out = lasagne.layers.get_output(network)
	deterministic_out = lasagne.layers.get_output(network, deterministic = True)

	st_normal_loss = T.mean(lasagne.objectives.categorical_crossentropy(stochastic_out,target_var))
	stochastic_loss = st_normal_loss + l2_regularization_rate * lasagne.regularization.l2(stochastic_out)

	dt_normal_loss = T.mean(lasagne.objectives.categorical_crossentropy(deterministic_out,target_var))
	deterministic_loss = dt_normal_loss

	all_params = lasagne.layers.get_all_params(network)

	updates = lasagne.updates.nesterov_momentum(
			stochastic_loss, all_params, learning_rate=learning_rate_alpha, momentum=mv)


	train_fn = theano.function([input_var, target_var],[stochastic_loss,stochastic_out], updates=updates)

	valid_fn = theano.function([input_var, target_var], [deterministic_loss, deterministic_out])


	batch_size = BATCH_SIZE

	# Finally, launch the training loop.
	print("Starting training...")
	# We iterate over epochs:
	train_loss_store = []
	val_loss_store = []
	val_acc_history = []
	train_acc_history = []
	best_valid_accuracy = 0.0
	best_model = None

	for epoch in range(num_epochs):
		# In each epoch, we do a full pass over the training data:
		train_err = 0
		train_batches = 0
		err = 0
		start_time = time.time()
		num_batches_train = int(np.ceil(len(X_train) / BATCH_SIZE))
		train_losses = []
		list_of_p = []
		for batch_num in range(num_batches_train):
			
			batch_slice = slice(batch_size * batch_num, batch_size * (batch_num + 1))
			X_batch = X_train[batch_slice]
			y_batch = y_train[batch_slice]

			loss,p_batch = train_fn(X_batch,y_batch)
			train_losses.append(loss)
			train_loss_store.append(loss)
			list_of_p.append(p_batch)
		
		train_loss = np.mean(train_losses)

		p_net = np.concatenate(list_of_p)
		predicted_classes = np.argmax(p_net, axis = 1)
		train_accuracy = sklearn.metrics.accuracy_score(y_train, predicted_classes)
		train_acc_history.append(train_accuracy)

		num_batches_valid = int(np.ceil(len(X_val) / batch_size))
		valid_losses = []
		list_of_p = []
		predicted_classes = []
		for batch_num in range(num_batches_valid):
			batch_slice = slice(batch_size * batch_num, batch_size * (batch_num + 1))
			X_batch = X_val[batch_slice]
			y_batch = y_val[batch_slice]

			loss,p_batch = valid_fn(X_batch,y_batch)
			valid_losses.append(loss)
			val_loss_store.append(loss)
			list_of_p.append(p_batch)

		valid_loss = np.mean(valid_losses)
		p_net = np.concatenate(list_of_p)
		predicted_classes = np.argmax(p_net,axis=1)
		valid_accuracy = sklearn.metrics.accuracy_score(y_val,predicted_classes)
		val_acc_history.append(valid_accuracy)
		
		total_time = time.time() - start_time

		# Print the results for this epoch:
		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, num_epochs, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_loss))
		print("  validation loss:\t\t{:.6f}".format(valid_loss))
		print("  validation accuracy:\t\t{:.2f} %".format(
			valid_accuracy * 100))
		print("  training accuracy:\t\t{:.2f} %".format(
			train_accuracy * 100))

		if valid_accuracy > best_valid_accuracy:
			best_valid_accuracy = valid_accuracy
			best_model = lasagne.layers.get_all_param_values(network)


		# if (epoch + 1) % epoch_delay[epoch_delay_count] == 0:
		#  	#learning_rate_alpha.set_value(np.float32(learning_rate_alpha.get_value() * learning_rate_decay))
		#  	learning_rate_alpha = learning_rate_alpha * learning_rate_decay
		#  	print(" Learning rate has been changed {}".format(learning_rate_alpha))	
		#  	epoch_delay_count = epoch_delay_count + 1;		
		
		# #train_loss_store.append(train_err/train_batches)
		#val_loss_store.append(val_err/val_batches)




	# After training, we compute and print the test error:
	## NOW LOAD THE BEST MODEL
	lasagne.layers.set_all_param_values(network,best_model)
	test_err = 0
	test_acc = 0
	test_batches = 0
	num_batches_test = int(np.ceil(len(X_test) / batch_size))
	test_losses = []
	list_of_p = []
	predicted_classes = []
	for batch_num in range(num_batches_test):
		batch_slice = slice(batch_size * batch_num, batch_size * (batch_num + 1))
		X_batch = X_test[batch_slice]
		y_batch = y_test[batch_slice]

		loss,p_batch = valid_fn(X_batch,y_batch)
		list_of_p.append(p_batch)

	p_net = np.concatenate(list_of_p)
	predicted_classes = np.argmax(p_net,axis=1)
	test_accuracy = sklearn.metrics.accuracy_score(y_test,predicted_classes)

	print("Final results:")
	print("  test accuracy:\t\t{:.2f} %".format(
		test_accuracy * 100))

	#file_name = 'all_net_model_a.npz'
	file_name = 'model_' + str(x)+ '_2_alpha_' + str(alpha) +'_momentum_' + str(mv) + '_l2_reg_' + str(l2_rate)+'_with_dropout_lt.npz'

	# Optionally, you could now dump the network weights to a file like this:
	if save==1:
		np.savez(file_name, *lasagne.layers.get_all_param_values(network))
	else:
		 with np.load('cifar_10_cnn2.npz') as f:
			param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		 lasagne.layers.set_all_param_values(network, param_values)

	
	#data_file_name = 'data_value_all_net_model_a.pckl'
	data_file_name = 'data_value_all_model_' +str(x) +'_2_alpha_'+ str(alpha) +'_momentum_' + str(mv) + '_l2_reg_' + str(l2_rate)+'_with_dropout_lt.pckl'

	f = open(data_file_name, 'w')
	pickle.dump([train_loss_store,val_loss_store,train_acc_history,val_acc_history], f)
	f.close()

	#
	# And load them again later on like this:
	# with np.load('model.npz') as f:
	#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	# lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
	if ('--help' in sys.argv) or ('-h' in sys.argv):
		print("Trains a neural network on MNIST using Lasagne.")
		print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
		print()
		print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
		print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
		print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
		print("       input dropout and DROP_HID hidden dropout,")
		print("       'cnn' for a simple Convolutional Neural Network (CNN).")
		print("EPOCHS: number of training epochs to perform (default: 500)")
	else:
		kwargs = {}
		if len(sys.argv) > 1:
			#kwargs['model'] = sys.argv[1]
			kwargs['x'] = int(sys.argv[1])
			kwargs['save'] = int(sys.argv[2])
		if len(sys.argv) > 3:
			kwargs['num_epochs'] = int(sys.argv[3])
		if len(sys.argv) > 4:
			kwargs['alpha'] = float(sys.argv[4])
		if len(sys.argv) > 5:
			kwargs['mv'] 	= float(sys.argv[5])
			kwargs['l2_rate'] = float(sys.argv[6])
					
		main(**kwargs)



