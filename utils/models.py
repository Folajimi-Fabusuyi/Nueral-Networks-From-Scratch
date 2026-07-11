import numpy as np
import math

class Loss:
    def CategoricalCrossEntropy(actual, predicted):
        return -np.sum(actual * np.log(predicted + 1e-9))
    
    def BinaryCrossEntropy(actual, predicted, count):
        pass

    def MeanSquaredError(actual, predicted, count):
        return np.sum(np.power(actual - predicted, 2)) * (1/count)
    
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
        if derivative == False:
            return 1 / (1 + np.exp(-1 * x))
        return x * (1 - x)
    
    def Tanh(x, derivative=False):
        if derivative == False:  
            return np.tanh(x)
        return 1 - np.power(x, 2)


class Settings:
    def __init__(self, input, output, hidden="16, 8", mini_batch='all', lr=0.001, nn_type="mp", activation="leaky_relu", output_activation="relu"):
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
        
    def predict(self, input):
        self.model.predict(input)


class MultilayerPerceptron:
    def __init__(self, settings: Settings):
        self.load(settings)

    def load(self, settings: Settings):
        self.input = np.array(settings.input).T 
        self.test_input = None
        self.true_output = np.array(settings.output)
        
        self.input_node_count = self.input.shape[0]
        self.hidden_node_count_per_layer = settings.hidden
        self.hidden_layers_count = len(self.hidden_node_count_per_layer)
        self.output_node_count = self.true_output.shape[0]
        
        self.epoch = 1
        self.epoch_ended = False
        
        if settings.mini_batch == "all": self.mini_batch = self.input.shape[1]
        else: self.mini_batch = settings.mini_batch
        self.batch_index = self.mini_batch
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
        self.loss = []
        self.avg_loss = 0
        
        self.initializeNetwork()
        
    def train(self):
        out = self.forwardPass(output=True)
        self.backPropagate()
        self.updatePass()
        if out is not None:
            print(f"Epoch {self.epoch} Loss: {self.avg_loss}\t Out: {out}")
        
    def predict(self, input, output=False):
        self.test_input = np.array(input)
        out = self.forwardPass(predicting=True, output=output)
        self.reset()
        print(f"Out: {out}")
    
    def initializeNetwork(self):            
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
    
    def forwardPass(self, predicting=False, output=False):
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
        
        if output or predicting:
            return self.output
        
    def backPropagate(self):
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
    # nn.train_model()
    while nn.model.avg_loss > 0.001 or nn.model.epoch == 1:
        nn.train_model()
        
    # while nn.model.epoch <= 20000:
    #     nn.train_model()

    nn.predict([[1], [0]])
    nn.predict([[1], [1]])
    nn.predict([[0], [0]])
    nn.predict([[1, 1], [0, 1]])
    
    # print(math.sqrt(18))
    # print(math.sqrt(125))
    # print(math.sqrt(32))
    # print(math.sqrt(49 + 9), math.sqrt(25+ 64))
    
    # [2.23606798, 2.82842712]
    
    # TODO: Implement mini batch and epoch logic next
    # TODO: Implement logging for networks
    # TODO: Make CTkinter with embedded graph and dashboard