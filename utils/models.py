import numpy as np

class Loss:
    def CategoricalCrossEntropy(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return -np.sum(actual * np.log(predicted + 1e-9))
    
    def BinaryCrossEntropy(actual: np.ndarray, predicted: np.ndarray, count: int) -> np.ndarray:
        pass

    def MeanSquaredError(actual: np.ndarray, predicted: np.ndarray, count: int) -> np.ndarray: 
        return np.sum(np.power(actual - predicted, 2)) * (1/count)
    
class Activations:
    # Output only activation
    def SoftMax(x: np.ndarray) -> np.ndarray:
        shifted_x = x - np.max(x)
        return np.exp(shifted_x) / np.sum(np.exp(shifted_x))
    
    # Either input or output activations
    def RelU(x: np.ndarray, derivative=False) -> np.ndarray:
        if derivative == False:
            return np.maximum(0, x)
        return np.where(x > 0, 1, 0)
    
    def LeakyRelU(x: np.ndarray, derivative=False) -> np.ndarray:
        if derivative == False:
            return np.where(x > 0, x, 0.01 * x) 
        return np.where(x > 0, 1, 0.01)
    
    def Sigmoid(x: np.ndarray, derivative=False) -> np.ndarray:       
        if derivative == False:
            return 1 / (1 + np.exp(-1 * x))
        return x * (1 - x)
    
    def Tanh(x: np.ndarray, derivative=False) -> np.ndarray:
        if derivative == False:  
            return np.tanh(x)
        return 1 - np.power(x, 2)


class Settings:
    ''' 
    Sets up settings for neural network
    
    Parameters:
    input (list[list[int]]): The input array, inputs are row first.
    output (list[list[int]]): The output array.
    hidden (str): Hidden layers and their nodes seperated by comma. "16, 8" means 16 node layer followed by 8 node layer.
    mini_batch (int): Amount of input to read through at a time. Defaults to all.
    lr (float): The learning rate.
    model_type (str): The type of model being created. Defaults to multilayer perceptron.
    activation (str): The hidden layer activations. Defaults to leaky_relu.
    output_activation(str): The output layer activations. Defaults to relu.
    '''
    
    def __init__(self, input: list[list[int]], output: list[list[int]], hidden="16, 8", mini_batch=0, lr=0.001, model_type="mp", activation="leaky_relu", output_activation="relu"):
        model_dict = {"mp": MultilayerPerceptron}
        activation_dict = {"relu": Activations.RelU,
                           "sigmoid": Activations.Sigmoid,
                           "softmax": Activations.SoftMax,
                           "tanh": Activations.Tanh,
                           "leaky_relu": Activations.LeakyRelU}
        
        self.input = input
        self.output = output
        self.hidden = [int(layer) for layer in hidden.split(",")]
        self.mini_batch = mini_batch
        
        self.lr = lr
        self.activation = activation_dict[activation]
        self.output_activation = activation_dict[output_activation]
        self.model_type = model_dict[model_type]
        
               
class NeuralNetwork:
    '''
    Creates a neural network based off of created settings
    
    Parameters:
    settings (Settings): The settings of the nueral network
    '''
    
    def __init__(self, settings: Settings):        
        self.settings = settings
        self.model = settings.model_type(settings)
        
    def train_model(self, output=False):
        '''Trains model for one epoch'''
        self.model.train(output)
        
    def predict(self, input: list[list[int]]) -> list[int]: # Change to return a list of integers
        '''Uses existing weights and biases to predict input'''
        return self.model.predict(input)


class MultilayerPerceptron:
    '''
    Neural network characterized by input, hidden and output layers only
    
    Parameters:
    settings (Settings): The settings of the nueral network
    '''
    
    
    def __init__(self, settings: Settings):
        self.load(settings)

    def load(self, settings: Settings):
        '''Loads settings and initializes model'''
        
        # Input is entered in transposed form for easier programatic input creation and then rotated back
        self.input = np.array(settings.input).T 
        self.test_input = np.array([])
        self.true_output = np.array(settings.output)
        
        self.input_node_count = self.input.shape[0]
        self.hidden_node_count_per_layer = settings.hidden
        self.hidden_layers_count = len(self.hidden_node_count_per_layer)
        self.output_node_count = self.true_output.shape[0]
        
        self.epoch = 1
        self.epoch_ended = False
        
        if settings.mini_batch == 0: self.mini_batch = self.input.shape[1]
        else: self.mini_batch = settings.mini_batch
        self.batch_index = self.mini_batch
        self.actual_batch = 0
        
        self.activation: Activations = settings.activation
        self.output_activation: Activations = settings.output_activation

        self.learning_rate = settings.lr
        
        self.raw_hidden_layers: list[np.ndarray] = []
        self.hidden_layers: list[np.ndarray] = []
        
        self.raw_output: np.ndarray = []
        self.output: np.ndarray = []
        
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        
        self.error: list[np.ndarray] = []
        self.gradient: list[np.ndarray] = []
        self.loss: list[int] = []
        self.avg_loss = 0
        
        self.initializeNetwork()
        
    def train(self, debug=False):
        '''Trains model for one epoch'''
        
        out = self.forwardPass()
        self.backPropagate()
        self.updatePass()
        if debug:
            print(f"Epoch {self.epoch} Loss: {self.avg_loss}\t Out: {out}")
        
    def predict(self, input: list[list[int]]) -> list[int]:
        '''Uses existing weights and biases to predict input'''
        
        self.test_input = np.array(input)
        out = self.forwardPass(predicting=True)
        self.reset()
        return out
    
    def initializeNetwork(self):     
        '''Aggregates weights and biases'''    
           
        self.weights.append(np.random.randn(self.hidden_node_count_per_layer[0], self.input_node_count) * 0.1) # Input node biases
        for layer in range(self.hidden_layers_count):
            if layer < self.hidden_layers_count - 1:
                self.weights.append(np.random.randn(self.hidden_node_count_per_layer[layer+1], self.hidden_node_count_per_layer[layer]) * 0.1)
                continue
            self.weights.append(np.random.randn(self.output_node_count, self.hidden_node_count_per_layer[layer]) * 0.1)
            
        left_bound, right_bound = 0, 1 # For bias randomization
        for layer in range(self.hidden_layers_count):
            self.biases.append(np.random.randn(self.hidden_node_count_per_layer[layer], 1) * 0.1)
        self.biases.append(np.random.randn(self.output_node_count, 1) * 0.1) # Output node biases
    
    def forwardPass(self, predicting=False) -> None | list[int]:
        '''Aggregates hidden and output layers'''
        
        if predicting: input = self.test_input
        else:
            input = self.input.T[self.batch_index - self.mini_batch: self.batch_index].T
        
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
        
        return self.output.tolist()[0]
        
    def backPropagate(self):
        '''Finds gradients and error to update weights and biases'''
        
        true_output = self.true_output.T[self.batch_index - self.mini_batch: self.batch_index].T
        
        if self.output_activation is Activations.SoftMax:
            self.loss.append(Loss.CategoricalCrossEntropy(true_output, self.output))
        else:
            self.loss.append(Loss.MeanSquaredError(true_output, self.output, self.actual_batch))
        
        
        if self.output_activation is Activations.SoftMax:
            output_error = (self.output - true_output)
        else:
            output_error = (self.output - true_output) * self.output_activation(self.output, derivative=True)
        self.error.append(output_error)
        
        for layer in range(len(self.weights) - 1, -1, -1):
            
            if layer > 0:
                layer_input = self.hidden_layers[layer - 1]
            else:
                layer_input = self.input.T[self.batch_index - self.mini_batch: self.batch_index].T
                
            weight_gradient = self.error[-1] @ layer_input.T
            self.gradient.append(weight_gradient)
            
            if layer > 0:
                passed_error = self.weights[layer].T @ self.error[-1]
                next_node_error = passed_error * self.activation(self.hidden_layers[layer - 1], derivative=True)
                self.error.append(next_node_error)
                      
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
        
        for index in range(len(self.weights)):
            self.weights[index] -= self.gradient[index] * self.learning_rate
            self.biases[index] -= np.sum(self.error[index], axis=1, keepdims=True) * self.learning_rate

        self.reset()
    
    def reset(self):
        self.hidden_layers = []
        self.output = []
        self.error = []
        self.gradient = []
        self.test_input = None
        
        
if __name__ == "__main__":
    
    nn_settings = Settings(
        input=[[1, 1], [0, 1], [1, 0], [0, 0]], 
        output=[[0, 1, 1, 0]],
        activation="tanh",
        output_activation="leaky_relu",
        # mini_batch=1,
        hidden="10, 5",
        lr=0.01
    )
    
    nn = NeuralNetwork(nn_settings)
    while nn.model.avg_loss > 0.0001 or nn.model.epoch == 1:
        nn.train_model(output=True)
        
    # while nn.model.epoch <= 20000:
    #     nn.train_model()

    user_input = "Y"
    while user_input.lower() not in ["q", "quit"]:
        user_input = input("Enter input for model prediction, separated by commas ['q' to quit]: ")
        if user_input in ["q", "quit"]:
            break
        data = [[int(number)] for number in user_input.split(",")]
        prediction = nn.predict(data)
        print(round(*prediction))

    nn.predict([[1], [1]])
    nn.predict([[0], [0]])
    nn.predict([[1, 1], [0, 1]])
    
    
    # TODO: Implement train_until conditional for model 
    # TODO: Implement logging for networks and potential graphs
    # TODO: Make CTkinter with embedded graph and dashboard