import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#import matplotlib.pyplot as plt

# Import data
print("Reading in data...")
data = pd.read_csv('data_stocks.csv')

# Drop date variable, just leaves stock prices
data = data.drop(['DATE'], 1)

# Dimensions of dataset
n = data.shape[0] # 41266 (number of values)
p = data.shape[1] # 501   (number of stocks)

# Make data a numpy array
data = data.values

print("Successful data read. " + str(p) + " stocks with " + str(n) + " points each")
print("-"*50)

# Training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
print("Training network with values 0 to " + str(train_end))
test_start = train_end + 1
test_end = n
print("Testing network with values " + str(test_start) + " to " + str(test_end))
print("-"*50)

# slices data
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

'''# Creates scaler for scaling input values from 0-1
print("Scaling data...")
scaler = MinMaxScaler()
scaler.fit(data_train)

# Transforms data from 0-1 using scaler
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
print("Data scaled")
print("-"*50)'''

# Build X and y
print("Building X and y vectors")
X_train = data_train[:, 1:] # input matrix of scaled stock values
print("X:")
print(X_train.shape)
y_train = data_train[:, 0]  # output vector
print("y:")
print(y_train.shape)
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# Initializer functions to start weights and biases
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Defines network layer parameters
n_stocks = 500
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1

# Placeholder vectors for input and output
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
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
batch_size = 256

for e in range(epochs):
    print("Epoch {} / {}".format(e,epochs))
    
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
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})



# Print final MSE after Training
'''print("Testing next value, expected = {}".format(y_test[0]))
prediction = net.run(out, feed_dict={X: X_test[0]})
print("Actual = {}".format(prediction))'''


