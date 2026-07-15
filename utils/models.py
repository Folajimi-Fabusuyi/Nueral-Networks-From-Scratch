import numpy as np
import os

from functions import Activations, Loss


class Settings:
    '''
    Sets up settings for neural network \n
    
    Parameters:\n
        input (list[list[int]]): The input array, inputs are row first.\n
        output (list[list[int]]): The output array.\n
        hidden (str): Hidden layers and their nodes seperated by comma. "16, 8" means 16 node layer followed by 8 node layer.\n
        mini_batch (int): Amount of input to read through at a time. Defaults to all inputs.\n
        lr (float): Rate at which model updates weights and biases. Defualts to 0.001\n
        momentum (float): Rate at which which model preserves velocity. Defaults to 0.9.\n
        model_type (str): The type of model being created. Defaults to multilayer perceptron.\n
        activation (str): The hidden layer activations. Defaults to leaky_relu.\n
        output_activation (str): The output layer activations. Defaults to relu.\n
        normalize (bool): Determines if network should automatically scale down inputs and outputs to smaller values. Defaults to False.\n
        norm (tuple[float, float]): Sets the value inputs and ouputs are scaled down by, to this value.\n
        dropout_rate (float): Percentage of random hidden layer nodes that are turned off during training forward passes. Defaults to 0.0.\n
    '''
    
    def __init__(self, input: list[list[int]], output: list[list[int]], hidden="16, 8", 
                 mini_batch=0, lr=0.001, momentum=0.9, model_type="mp", 
                 activation="leaky_relu", output_activation="relu", normalize=False, norm: tuple[float, float]=(0.0, 0.0),
                 dropout_rate=0.0, train_split=1.0):
        model_dict = {"mp": MultilayerPerceptron}
        activation_dict = {"relu": Activations.RelU,
                           "sigmoid": Activations.Sigmoid,
                           "softmax": Activations.SoftMax,
                           "tanh": Activations.Tanh,
                           "leaky_relu": Activations.LeakyRelU,
                           "linear": Activations.Linear}
        
        # Train-Test Split
        
        
        # Transpose back to correctly shaped array
        self.input = np.array(input).T
        self.output = np.array(output).T
        
        self.normalize = normalize
        if norm[0] > 0 and norm[1] > 0: # Manual norm values, useful when input and output set has a guaranteed limit, aka, pixel color.
            self.normalize = True
            self.i_scaler = norm[0]
            self.o_scaler = norm[1]
        elif self.normalize: # Automatic norm value, uses max input and output values            
            self.i_scaler = max((self.input * -1).max(), self.input.max())
            self.o_scaler = max((self.output * -1).max(), self.output.max())
            
            self.i_scaler = max(1e-2, float(self.i_scaler))
            self.o_scaler = max(1e-2, float(self.o_scaler))
                 
            self.input /= self.i_scaler
            self.output /= self.o_scaler
            
        
        self.dropout_rate = max(0.0, min(dropout_rate, 0.99))
        self.hidden = [int(layer) for layer in hidden.split(",")]
        self.mini_batch = mini_batch
        
        self.lr = lr
        self.friction = momentum
        self.activation = activation_dict[activation]
        self.output_activation = activation_dict[output_activation]
        self.model_type = model_dict[model_type]
            
        
               
class NeuralNetwork:
    '''
    Creates a neural network based off of created settings\n
    
    Parameters:\n
        settings (Settings): The settings of the nueral network\n
    '''
    
    def __init__(self, settings: Settings):        
        self.settings = settings
        self.model = settings.model_type(settings)
        
    def train_model(self, debug=False):
        '''Trains model for one epoch'''
        self.model.train(debug)
        
    def predict(self, input: list[list[int]]) -> list[int]: # Change to return a list of integers
        '''Uses existing weights and biases to predict input'''
        return self.model.predict(input)


class MultilayerPerceptron:
    '''
    Neural network characterized by input, hidden and output layers only\n
    
    Parameters:\n
        settings (Settings): The settings of the nueral network\n
    '''
    
    
    def __init__(self, settings: Settings):
        self.load(settings)

    def load(self, settings: Settings):
        '''Loads settings and initializes model'''
        
        # Input/Output are entered in transposed form for easier programatic input creation and then rotated back
        self.input = settings.input
        self.test_input = np.array([])
        self.true_output = settings.output
        
        # Node counts
        self.input_node_count = self.input.shape[0]
        self.hidden_node_count_per_layer = settings.hidden
        self.hidden_layers_count = len(self.hidden_node_count_per_layer)
        self.output_node_count = self.true_output.shape[0]
        
        # Epoch variables
        self.epoch = 1
        self.epoch_ended = False
        
        # Batch variables
        if settings.mini_batch == 0: self.mini_batch = self.input.shape[1]
        else: self.mini_batch = settings.mini_batch
        self.batch_index = self.mini_batch
        self.actual_batch = 0
        
        self.dropout_rate = settings.dropout_rate
        self.dropout_masks: list[np.ndarray] = []
        
        # Activation functions
        self.activation: Activations = settings.activation
        self.output_activation: Activations = settings.output_activation

        # Backpropagation variables
        self.learning_rate = settings.lr
        
        self.raw_hidden_layers: list[np.ndarray] = []
        self.hidden_layers: list[np.ndarray] = []
        
        self.raw_output: np.ndarray = []
        self.output: np.ndarray = []
        
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        
        self.velocity: list[np.ndarray] = []
        self.friction = settings.friction
        
        self.error: list[np.ndarray] = []
        self.gradient: list[np.ndarray] = []
        self.loss: list[int] = []
        self.avg_loss = 0
        
        # Input and output normalization scalars, for stabilizing network for larger numbers
        if settings.normalize:
            self.i_scaler = settings.i_scaler
            self.o_scaler = settings.o_scaler
        
        self.initializeNetwork()
        
    def train(self, debug=False):
        '''Trains model for one epoch'''
        
        out = self.forwardPass().flatten().tolist()
        self.backPropagate()
        self.updatePass()
        if debug:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"Epoch {self.epoch} | Loss: {round(self.avg_loss, 10)}")
            print(f"Lr: {round(self.learning_rate, 10)} | Calc_Avg_L: {round(np.mean(self.loss), 10)}")
            print(f"Current_Batch: {len(self.loss)} | Actual_Batch: {self.actual_batch} | Equal: {len(self.loss) == self.actual_batch - 1}")
        
    def predict(self, input: list[list[int]]) -> list[int]:
        '''Uses existing weights and biases to predict input'''

        self.test_input = np.array(input).T / self.i_scaler # Normalizes input to match size of training input
        out = self.forwardPass(predicting=True)
        out = (out * self.o_scaler).flatten().tolist()
        
        self.reset()
        return out
    
    def initializeNetwork(self):     
        '''Aggregates weights and biases'''    
           
        # Weight aggregation
        self.weights.append(np.random.randn(self.hidden_node_count_per_layer[0], self.input_node_count) * 0.1) # First layer weights
        for layer in range(self.hidden_layers_count):
            if layer < self.hidden_layers_count - 1:
                self.weights.append(np.random.randn(self.hidden_node_count_per_layer[layer+1], self.hidden_node_count_per_layer[layer]) * 0.1)
                continue
            self.weights.append(np.random.randn(self.output_node_count, self.hidden_node_count_per_layer[layer]) * 0.1)
            
        # Hidden layers' randomized biases are lowered to reduce inital network noise
        for layer in range(self.hidden_layers_count):
            self.biases.append(np.random.randn(self.hidden_node_count_per_layer[layer], 1) * 0.01)
        
        # Output biases are set to 0 to reduce initial network noise as well
        self.biases.append(np.zeros((self.output_node_count, 1)))
    
    def forwardPass(self, predicting=False) -> None | np.ndarray:
        '''Aggregates hidden and output layers'''
        
        if predicting: input = self.test_input
        else:
            input = self.input.T[self.batch_index - self.mini_batch: self.batch_index].T

        self.actual_batch = input.shape[1]

        # Hidden layer propagation
        for index in range(self.hidden_layers_count):                         
            if index == 0:
                raw_hidden_layer = self.weights[index] @ input + self.biases[index]
            else:
                raw_hidden_layer = (self.weights[index] @ self.hidden_layers[index - 1] + self.biases[index])
            activated_hidden_layer = self.activation(raw_hidden_layer)

            # Dropout logic
            if not predicting:
                mask = (np.random.rand(*activated_hidden_layer.shape) < (1 - self.dropout_rate)) / (1 - self.dropout_rate)
                self.dropout_masks.append(mask)
                activated_hidden_layer = activated_hidden_layer * mask
                
            self.hidden_layers.append(activated_hidden_layer)
                
        # Output layer propagation
        raw_output = self.weights[-1] @ self.hidden_layers[-1] + self.biases[-1]
        self.output = self.output_activation(raw_output)
        
        return self.output
        
    def backPropagate(self):
        '''Finds gradients and error to update weights and biases'''
        
        true_output = self.true_output.T[self.batch_index - self.mini_batch: self.batch_index].T
        
        # This calculation here is really only for us to view how the model is doing
        # Might possibly be useful for more advanced learing rate adjustments
        if self.output_activation is Activations.SoftMax:
            self.loss.append(Loss.CategoricalCrossEntropy(true_output, self.output))
        else:
            self.loss.append(Loss.MeanSquaredError(true_output, self.output, self.actual_batch))
        
        # The derivative of the loss
        if self.output_activation in [Activations.SoftMax, Activations.Linear]:
            output_error = (self.output - true_output)
        else:
            output_error = (self.output - true_output) * self.output_activation(self.output, derivative=True)
        self.error.append(output_error)
        
        # Gradient and error propagation
        for layer in range(len(self.weights) - 1, -1, -1):           
            if layer > 0:
                layer_input = self.hidden_layers[layer - 1]
            else:
                layer_input = self.input.T[self.batch_index - self.mini_batch: self.batch_index].T
                
            weight_gradient = self.error[-1] @ layer_input.T
            self.gradient.append(weight_gradient)
            
            if layer > 0:
                passed_error = self.weights[layer].T @ self.error[-1]
                activation_derivative = self.activation(self.hidden_layers[layer - 1], derivative=True)
                
                # Account for dropouts
                mask = self.dropout_masks[layer - 1]
                next_node_error = passed_error * activation_derivative * mask

                self.error.append(next_node_error)
        
        # Epoch and batch logic           
        if self.batch_index < self.input.shape[1]:
            self.batch_index += self.mini_batch
            self.epoch_ended = False
        else:
            # This means that an epoch has ended
            self.epoch += 1
            self.epoch_ended = True
            self.batch_index = self.mini_batch
            self.avg_loss = np.mean(self.loss)
            self.loss = []
            
        
    def updatePass(self):
        '''Updates weights and biases by gradient'''

        self.gradient.reverse()
        self.error.reverse()
        
        # Momentum implementation
        if self.velocity == []:
            self.velocity = [np.zeros(self.weights[i].shape) for i in range(len(self.weights))]
        self.velocity = [(self.friction * self.velocity[i]) - (self.learning_rate * self.gradient[i]) for i in range(len(self.weights))]
        
        # Scheduled Learning Decay
        if self.epoch % 100 == 0 and len(self.loss) == 0:
            self.learning_rate *= 0.9

        # Weight and bias nudges
        for index in range(len(self.weights)):
            # self.weights[index] -= self.gradient[index] * self.learning_rate
            self.weights[index] += self.velocity[index]
            self.biases[index] -= np.sum(self.error[index], axis=1, keepdims=True) * self.learning_rate

        self.reset()
    
    def reset(self):
        self.hidden_layers = []
        self.output = []
        self.error = []
        self.gradient = []
        self.dropout_masks = []
        self.test_input = None
        
        
if __name__ == "__main__":
    import random
    import math
    
    pythagorean_in = []
    pythagorean_out = []
    
    samples_per_bracket = 250
    brackets = [(0, 5), (5, 20), (20, 50), (50, 100)]
    
    for lower_bound, upper_bound in brackets:
        for _ in range(samples_per_bracket):
            # Generate floats instead of ints for smoother curves!
            i = random.uniform(lower_bound, upper_bound)
            j = random.uniform(lower_bound, upper_bound)
            
            k = math.sqrt(math.pow(i, 2) + math.pow(j, 2))
            
            pythagorean_in.append([i, j])
            pythagorean_out.append([k])
            
    for _ in range(10):
        pythagorean_in.append([0.0, 0.0])
        pythagorean_out.append([0.0])
    
    indices = list(range(len(pythagorean_in)))
    random.shuffle(indices)
    
    pythagorean_in = [pythagorean_in[i] for i in indices]
    pythagorean_out = [pythagorean_out[i] for i in indices]
    
    x_or_in = [[1, 1], [0, 1], [1, 0], [0, 0]]
    x_or_out = [[0], [1], [1], [0]]
    
    nn_settings = Settings(
        input= pythagorean_in, 
        output= pythagorean_out,
        activation="leaky_relu",
        output_activation="linear", 
        mini_batch=64,
        hidden="32",              
        normalize=True,
        # dropout_rate=0.2,
        lr=1e-3
    )
    
    # Training loop
    nn = NeuralNetwork(nn_settings)
    while nn.model.avg_loss > 1e-6 or nn.model.epoch == 1:
        if nn.model.epoch % 100 == 0 and len(nn.model.loss) == 0:
            leave = input("Quit[y/n]: ")
            if leave == "y":
                break
        nn.train_model(debug=True)


    user_input = ""
    while user_input.lower() not in ["q", "quit"]:
        user_input = input("Enter input for model prediction, separated by commas ['q' to quit]: ")
        if user_input in ["q", "quit"]:
            break
        data = [[float(number) for number in user_input.split(",")]]
        prediction = nn.predict(data)[0]
        print(round(prediction, 2))
        print(f"Avg: {nn.model.true_output.mean()}")

    
    # DONE: Add functionality for scaled input during prediction
    # DONE: Add Learning rate adapting
        # NOTE: Could definitely improve on this
    # DONE: Add Dropout during forward pass
    # DONE: Standardize input and output structure for passing into model
    
    
    # TODO: Implement validation loss each epoch with train-test split
    # TODO: Implement train_until conditional for model 
    # TODO: Implement model saving
    # TODO: Implement dynamic model setting changes
    
    # TODO: Implement logging for networks and potential graphs
    # TODO: Make PySide6 with embedded graph and dashboard