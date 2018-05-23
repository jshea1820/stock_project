#!/anaconda3/bin/python3

'''

Jack Shea
23 May, 2018

price_predictor.py

- Creates and trains a neural network for determining
- s&p index price based on changes in constituent stocks

'''

# Imported modules
import pandas as pd
import tensorflow as tf
import numpy as np

# Brings in input training data as dataframe
data = pd.read_csv('data_update.csv', index_col = 'Date')
print(data)

# Creates input matrix and output vector
n = data.shape[0]
p = data.shape[1] - 1
print("Data is of shape ({},{})".format(n,p))

# Turns data into np array
data = data.values

# Divides data into train and test set
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]
print("Training from 0 to {}, testing from {} to {}".format(train_end-1, test_start, n))

# Creates training and testing inputs and outputs
X_train = data_train[:, :-1]
y_train = data_train[:, -1]
X_test = data_test[:, :-1]
y_test = data_test[:, -1]

# Creates placeholders for input matrix and output vector
X = tf.placeholder(dtype=tf.float32, shape=[None, p])
y = tf.placeholder(dtype=tf.float32, shape=[None])

# Model architecture parameters
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([p, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))

# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))
mse = tf.reduce_mean(tf.squared_difference(out, y))

opt = tf.train.AdamOptimizer().minimize(mse)

# Make Session
net = tf.Session()

# Run initializer
net.run(tf.global_variables_initializer())

# Number of epochs and batch size
epochs = 10
batch_size = 32

for e in range(epochs):
    print("Epoch {}".format(e))

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]

        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, y: batch_y})


pred = float(net.run(out, feed_dict={X: X_test[0:1]}))
print("Prediction = {}".format(pred))
print("Actual = {}".format(y_test[0]))














