'''
Jack Shea
neural_net_model_generator.py

Class for stock analysis and neural net generation using quandl
and tensorflow
'''


import quandl
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# sample stock
stock = 'AMZN'

# Reads in stock data
print("Reading in data for stock {}".format(stock))
stock = quandl.get('WIKI/' + stock)
print("Stock successfully read in")
num_days = len(stock['Open'])
print("{} days worth of data".format(num_days))

# Calculates daily change percentage values
stock['Delta_p'] = (stock['Close'] - stock['Open']) / stock['Open']
print("Calculated Daily % changes")

# how many historical points will be used to calculate the next day's change
hist_size = 500

# Input matrix with output vector
x = np.ndarray(shape = (num_days - hist_size - 1, hist_size))
y = np.ndarray(shape = (num_days - hist_size - 1, ))

n = x.shape[0]  # number of input data points
p = x.shape[1]  # size of each data point

# Populates input matrix and output vector
for i in range(num_days - hist_size - 1):

    # Checkpoint print statement
    if i % 100 == 0:
        print("{} days completed".format(i))

    # Populates row with 500 consective change percentage values
    for j in range(hist_size):
        x[i, j] = stock['Delta_p'][i+j]

    # Populates output vector with % change value for the day after the
    # 500 consecutive days
    y[i] = stock['Delta_p'][hist_size + i]


# Training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
print("Training network with values 0 to " + str(train_end))
test_start = train_end + 1
test_end = n
print("Testing network with values " + str(test_start) + " to " + str(test_end))
print("-"*50)

# Creates training and testing input matrices and output vectors
X_train = x[np.arange(train_start, train_end), :]
X_test = x[np.arange(test_start, test_end), :]
y_train = y[np.arange(train_start, train_end)]
y_test = y[np.arange(test_start, test_end)]

# Initializer functions to start weights and biases
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Defines network layer parameters
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1

# Placeholder vectors for input and output
X = tf.placeholder(dtype=tf.float32, shape=[None, hist_size])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([hist_size, n_neurons_1]))
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

# Defines connections between layers
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Make Session
net = tf.Session()

# Run initializer
net.run(tf.global_variables_initializer())

# Number of epochs and batch size
epochs = 10
batch_size = 128

print("Network built, testing...")
for e in range(epochs):
    print("Epoch {}".format(e))
    
    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        print("Feeding batch {}".format(i))
        
        start = i * batch_size
        batch_x = X_train[start : (start + batch_size)]
        batch_y = y_train[start : (start + batch_size)]

        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})



# Print final MSE after Training
mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
print(mse_final)
    

