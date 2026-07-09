import numpy as np
import math


class Activations:
    # Output only activation
    def SoftMax(x):
        shifted_x = x - np.max(x)
        return np.exp(shifted_x) / np.sum(np.exp(shifted_x))
    
    # Either input or output activations
    def RelU(x, derivative=False):
        if derivative == False:
            return np.maximum(0, x)
        return np.where(x > 0, 1, 0)
    
    def LeakyRelU(x, derivative=False):
        if derivative == False:
            return np.where(x > 0, x, 0.01 * x) 
        return np.where(x > 0, 1, 0.01)
    
    def Sigmoid(x, derivative=False):
        def sigmoid(arr):
            return 1 / (1 + np.exp(-1 * arr))
        
        if derivative == False:
            return sigmoid(x)
        return sigmoid(x) * (1 - sigmoid(x))
    
    def Tanh(x, derivative=False):
        return (2 / (1 + np.exp(-2 / x))) + 1


class Settings:
    def __init__(self, input, output, hidden="16, 8, 4", mini_batch=1, lr=0.001, nn_type="mp", activation="leaky_relu", output_activation="relu"):
        self.input = input
        self.output = output
        self.hidden = [int(layer) for layer in hidden.split(",")]
        self.mini_batch = mini_batch
        
        self.lr = lr
        self.activation = activation
        self.output_activation = output_activation
        self.nn_type = nn_type
        
               
class NeuralNetwork:
    def __init__(self, settings: Settings):
        model_dict = {"mp": MultilayerPerceptron}
        activation_dict = {"relu": Activations.RelU,
                           "sigmoid": Activations.Sigmoid,
                           "softmax": Activations.SoftMax,
                           "tanh": Activations.Tanh,
                           "leaky_relu": Activations.LeakyRelU}
                
        settings.activation = activation_dict[settings.activation]
        settings.output_activation = activation_dict[settings.output_activation]
        
        self.settings = settings
        self.model = model_dict[settings.nn_type](settings)
        
    def train_model(self):
        self.model.train()
        
    def test_model(self, input):
        self.model.test(input)


class MultilayerPerceptron:
    def __init__(self, settings: Settings):
        self.load(settings)

    def load(self, settings: Settings):
        self.input = np.array(settings.input)
        self.test_input = None
        self.true_output = np.array(settings.output)
        
        self.input_node_count = self.input.shape[0]
        self.hidden_node_count_per_layer = settings.hidden
        self.hidden_layers_count = len(self.hidden_node_count_per_layer)
        self.output_node_count = self.true_output.shape[0]
        
        self.mini_batch = settings.mini_batch
        self.actual_batch = None
        
        self.activation = settings.activation
        self.output_activation = settings.output_activation

        self.learning_rate = settings.lr
        
        self.raw_hidden_layers = []
        self.hidden_layers = []
        
        self.raw_output = []
        self.output = []
        
        self.weights = []
        self.biases = []
        
        self.error = []
        self.gradient = []
        self.loss = 1
        
        self.initializeNetwork()
        
    def train(self):
        out = self.forwardPass(output=True)
        self.backPropagate()
        self.updatePass()
        print(f"Loss: {self.loss}\t Out: {out}")
        
    def test(self, input):
        self.test_input = np.array(input)
        out = self.forwardPass(test=True, output=True)
        self.reset()
        print(f"Out: {out}")
    
    def initializeNetwork(self):            
        self.weights.append(np.random.rand(self.hidden_node_count_per_layer[0], self.input_node_count)) # Input node biases
        for layer in range(self.hidden_layers_count):
            if layer < self.hidden_layers_count - 1:
                self.weights.append(np.random.rand(self.hidden_node_count_per_layer[layer+1], self.hidden_node_count_per_layer[layer]))
                continue
            self.weights.append(np.random.rand(self.output_node_count, self.hidden_node_count_per_layer[layer]))
            
        left_bound, right_bound = 0, 1 # For bias randomization
        for layer in range(self.hidden_layers_count):
            self.biases.append(np.random.random_sample((self.hidden_node_count_per_layer[layer], 1)) * (left_bound + right_bound) - left_bound)
        self.biases.append(np.random.rand(self.output_node_count, 1)) # Output node biases
    
    def forwardPass(self, test=False, output=False):
        if test: input = self.test_input
        else: input = self.input
        
        self.actual_batch = input.shape[1]
                
        # Hidden layer propagation
        for index in range(self.hidden_layers_count):             
            if index == 0:
                self.hidden_layers.append(self.activation(self.weights[index] @ input + self.biases[index]))
                continue
            raw_hidden_layer = (self.weights[index] @ self.hidden_layers[index - 1] + self.biases[index])
            self.hidden_layers.append(self.activation(raw_hidden_layer))
            
        # Output layer propagation
        raw_output = self.weights[-1] @ self.hidden_layers[-1] + self.biases[-1]
        self.output = self.output_activation(raw_output)
        
        if output:
            return self.output
        
    def backPropagate(self):
        if self.output_activation is Activations.SoftMax:
            self.loss = -np.sum(self.true_output * np.log(self.output + 1e-9))
        else:
            self.loss = np.sum(np.power(self.true_output - self.output, 2)) * (1/self.actual_batch)
        
        
        if self.output_activation is Activations.SoftMax:
            output_error = (self.output - self.true_output)
        else:
            output_error = (self.output - self.true_output) * self.output_activation(self.output, derivative=True)
        self.error.append(output_error)
        
        for layer in range(len(self.weights) - 1, -1, -1):
            
            if layer > 0:
                layer_input = self.hidden_layers[layer - 1]
            else:
                layer_input = self.input
                
            weight_gradient = self.error[-1] @ layer_input.T
            self.gradient.append(weight_gradient)
            
            if layer > 0:
                passed_error = self.weights[layer].T @ self.error[-1]
                next_node_error = passed_error * self.activation(self.hidden_layers[layer - 1], derivative=True)
                self.error.append(next_node_error)
            
        
    def updatePass(self):
        self.gradient.reverse()
        self.error.reverse()
        
        for index in range(len(self.weights)):
            self.weights[index] -= self.gradient[index] * self.learning_rate
            self.biases[index] -= np.sum(self.error[index], axis=1, keepdims=True) * self.learning_rate

        self.reset()
    
    def reset(self):
        self.hidden_layers = []
        self.output = []
        self.error = []
        
        
if __name__ == "__main__":
    
    nn_settings = Settings(
        input=[[1, 0, 0, 1], 
               [1, 1, 0, 0]], 
        output=[[0, 1, 0, 1]],
        activation="leaky_relu",
        output_activation="sigmoid",
        mini_batch=4,
        hidden="10, 10",
        lr=0.01
    )
    nn = NeuralNetwork(nn_settings)
    while nn.model.loss > 0.0001:
        nn.train_model()
        
    nn.test_model([[0], [0]])
    nn.test_model([[1], [0]])
    nn.test_model([[0], [1]])
    nn.test_model([[1, 0], [1, 0]])
    # [2.23606798, 2.82842712]