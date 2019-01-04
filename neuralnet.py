'''

neuralnet.py
- Implementation of neuralnet class

x = Neuralnet(layer_params) -> builds network architecture for network
based on list of hidden layer sizes

x.train(X_train, y_train) -> trains network with given input data

x.run(x_test) -> takes in input vector and outputs output vector


'''

class Neuralnet:
    ''' Class for easily implementing and testing neural networks '''
    
    def __init__(self, layer_params):
        ''' Creates network architecture from list of layer sizes '''

        # Sets some hardcoded network parameters
        sigma = 1
        weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
        bias_initializer = tf.zeros_initializer()

        # Placeholder variables for network input and output                                                          
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, layer_params[0]])
        self.y = self.Y = tf.placeholder(dtype=tf.float32, shape=[None])

        # Lists to hold the weight and bias layers
        weight_layers = []
        bias_layers = []                                                          
                                                             





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

