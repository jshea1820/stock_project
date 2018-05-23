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

# Print final MSE after Training
mse_final = net.run(mse, feed_dict={X: X_test, y: y_test})
print(mse_final)






'''
        
    def make_model_IO(self, hist_size):
        Makes the input matrix and output vector to prepare for training and testing

        # Keeps track of hist_size variable
        self.hist_size = hist_size
        
        # Input matrix with output vector
        self.x = np.ndarray(shape = (self.num_days - hist_size - 1, hist_size))
        self.y = np.ndarray(shape = (self.num_days - hist_size - 1, ))

        n = self.x.shape[0]  # number of input data points
        p = self.x.shape[1]  # size of each data point

        # Populates input matrix and output vector
        for i in range(self.num_days - hist_size - 1):

            # Checkpoint print statement
            if i % 500 == 0:
                print("{} days completed".format(i))

            # Populates row with hist_size consective change percentage values
            for j in range(hist_size):
                self.x[i, j] = self.stock_data['Delta_p'][i+j]

            # Populates output vector with % change value for the day after the
            # 500 consecutive days
            self.y[i] = self.stock_data['Delta_p'][hist_size+i]


        # Training and test data
        train_start = 0
        train_end = int(np.floor(0.8*n))
        print("Training network with values 0 to " + str(train_end))
        test_start = train_end + 1
        test_end = n
        print("Testing network with values " + str(test_start) + " to " + str(test_end))
        print("-"*50)

        # Creates training and testing input matrices and output vectors
        self.X_train = self.x[np.arange(train_start, train_end), :]
        self.X_test = self.x[np.arange(test_start, test_end), :]
        self.y_train = self.y[np.arange(train_start, train_end)]
        self.y_test = self.y[np.arange(test_start, test_end)]

        
    def create_network(self, layer_params):

        print("Creating network architecture: ")
        print(layer_params)
        
        # Initializer functions to start weights and biases
        sigma = 1
        weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
        bias_initializer = tf.zeros_initializer()
        
        # Placeholder vectors for input and output
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.hist_size])
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None])

        # Lists to hold the weight and bias layers
        weight_layers = []
        bias_layers = []
        
        # Layer 1: Variables for hidden weights and biases
        print("Making input layer from {} to {}".format(self.hist_size, layer_params[0]))
        weight_layers.append(tf.Variable(weight_initializer([self.hist_size, layer_params[0]])))
        bias_layers.append(tf.Variable(bias_initializer([layer_params[0]])))

        # Sets the next i layers
        for i in range(0,len(layer_params) - 1):
            print("Making layer from {} to {}".format(layer_params[i], layer_params[i+1]))
            weight_layers.append(tf.Variable(weight_initializer([layer_params[i], layer_params[i+1]])))
            bias_layers.append(tf.Variable(bias_initializer([layer_params[i+1]])))

        # Output layer: Variables for output weights and biases
        print("Making output layer from {} to {}".format(layer_params[-1], 1))
        W_out = tf.Variable(weight_initializer([layer_params[-1], 1]))
        bias_out = tf.Variable(bias_initializer([1]))

        # List to hold connections between layers
        hidden_layers = []

        # Appends first layer
        print("Connecting input layer to layer 1")
        hidden_layers.append(tf.nn.relu(tf.add(tf.matmul(self.X, weight_layers[0]), bias_layers[0])))

        # Appends next i layers
        for i in range(0, len(layer_params) - 1):
            print("Connecting layer {} to layer {}".format(i+1, i+2))
            hidden_layers.append(tf.nn.relu(tf.add(tf.matmul(hidden_layers[i], weight_layers[i+1]), bias_layers[i+1])))
        
        # Output layer (must be transposed)
        print("Connecting layer {} to output layer".format(len(layer_params)))
        self.out = tf.transpose(tf.add(tf.matmul(hidden_layers[-1], W_out), bias_out))
        
        # Cost function
        self.mse = tf.reduce_mean(tf.squared_difference(self.out, self.Y))

        # Optimizer
        self.opt = tf.train.AdamOptimizer().minimize(self.mse)

        
    def train_network(self, epoch_count, batch_size):
        print("Training network...")
        
        self.net = tf.Session()
        self.net.run(tf.global_variables_initializer())
            
        # Number of epochs and batch size
        epochs = epoch_count
        batch_size = 256
        
        for e in range(epochs):

            print("Epock {} of {}".format(e, epochs))
            
            # Shuffle training data
            shuffle_indices = np.random.permutation(np.arange(len(self.y_train)))
            random_X_train = self.X_train[shuffle_indices]
            random_y_train = self.y_train[shuffle_indices]

            # Minibatch training
            for i in range(0, len(self.y_train) // batch_size):
                start = i * batch_size
                batch_x = random_X_train[start:start + batch_size]
                batch_y = random_y_train[start:start + batch_size]

                # Run optimizer with batch
                self.net.run(self.opt, feed_dict={self.X: batch_x, self.Y: batch_y})

            

    def get_predictions(self, future_days):
        Returns an array of predictions for the next <future_days> days

        print("Retrieving predictions for the next {} days...".format(future_days))
        predictions = []
        history = []
        history.append([])
        for i in range(self.hist_size):
            history[0].append(self.x[-1][i])
              
        for i in range(future_days):
            predictions.append(float(self.net.run(self.out, feed_dict={self.X: history[i:i+1]})))
            history.append([])
            for j in range(1,self.hist_size):
                history[i+1].append(history[i][j])
            history[i+1].append(predictions[i])

        prediction_vals = []
        curr_value = self.stock_data['Open'][-1]
        prediction_vals.append(curr_value)

        for i in range(future_days):
            curr_value = curr_value + curr_value * predictions[i]
            prediction_vals.append(curr_value)

        return prediction_vals
        

        



if __name__ == "__main__":
    stock = Stocknet('AMZN')
    stock.make_model_IO(30)
    stock.create_network([1024, 512, 256, 128])
    stock.train_network(10, 256)
    predictions = stock.get_predictions(50)
    
    for i in range(len(predictions)):
        print("{} days from now, price will be {}".format(i, predictions[i]))'''
    
    
    
    
