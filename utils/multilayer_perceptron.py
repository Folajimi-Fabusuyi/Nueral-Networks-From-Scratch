import numpy as np
import math

template = {
    "input": {
        "arr": [],
        "n_nodes": "int",
        "batch": "int"
    },
    "hidden": {
        "n_layers": "int",
        "n_nodes": []
    },
    "output": {
        "n_out": "int"
    },
    "actual": [],
    "activation": "function",
    "output_activation": "function",
    "loss": "function",
    "learning_rate": "float"
}

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
    
    def Sigmoid(x, derivative=False):
        def sigmoid(arr):
            return 1 / (1 + np.exp(-1 * arr))
        
        if derivative == False:
            return sigmoid(x)
        return sigmoid(x) * (1 - sigmoid(x))

class MultilayerPerceptron:
    def __init__(self, blueprint):
        self.load(blueprint)

    def load(self, blueprint):
        self.input_node_count = blueprint["input"]["n_nodes"]
        self.hidden_node_count_per_layer = blueprint["hidden"]["n_nodes"]
        self.hidden_layers_count = blueprint["hidden"]["n_layers"]
        self.output_node_count = blueprint["output"]["n_out"]
        self.mini_batch_count = blueprint["input"]["n_batch"]
        
        self.activation_function = blueprint["activation"]
        self.output_activation_function = blueprint["output_activation"]
        self.loss_function = blueprint["loss"]
        self.true_output = np.array(blueprint["actual"])
        self.learning_rate = blueprint["learning_rate"]
        
        self.inputs = np.array(blueprint["input"]["arr"])
        
        self.raw_hidden_layers = []
        self.hidden_layers = []
        
        self.raw_output = []
        self.output = []
        
        self.weights = []
        self.biases = []
        
        self.error = []
        self.gradient = []
        self.loss = 0
        
        self.initializeNetwork()
        

    def train(self):
        out = self.forwardPass(output=True)
        self.backPropagate()
        self.updatePass()
        print(f"Loss: {self.loss}\t Out: {out}")

    def test(self):
        pass
    
    def initializeNetwork(self):     
        self.reset() 
        self.weights = []
        self.biases = []
        
        self.weights.append(np.random.rand(self.hidden_node_count_per_layer[0], self.input_node_count)) # Input node biases
        for layer in range(self.hidden_layers_count):
            if layer < self.hidden_layers_count - 1:
                self.weights.append(np.random.rand(self.hidden_node_count_per_layer[layer+1], self.hidden_node_count_per_layer[layer]))
                continue
            self.weights.append(np.random.rand(self.output_node_count, self.hidden_node_count_per_layer[layer]))
            
        left_bound, right_bound = 0, 1 # For bias randomization
        for layer in range(self.hidden_layers_count):
            self.biases.append(np.random.random_sample((self.hidden_node_count_per_layer[layer], 1)) * (left_bound + right_bound) - left_bound)
        self.biases.append(np.random.rand(self.output_node_count, self.mini_batch_count)) # Output node biases
    
    def forwardPass(self, output=False):        
        # Hidden layer propagation
        for index in range(self.hidden_layers_count):             
            if index == 0:
                self.hidden_layers.append(self.activation_function(self.weights[index] @ self.inputs + self.biases[index]))
                continue
            raw_hidden_layer = (self.weights[index] @ self.hidden_layers[index - 1] + self.biases[index])
            self.hidden_layers.append(self.activation_function(raw_hidden_layer))
            
        # Output layer propagation
        raw_output = self.weights[-1] @ self.hidden_layers[-1] + self.biases[-1]
        self.output = self.output_activation_function(raw_output)
        
        if output:
            return self.output
        
    def backPropagate(self):
        if self.output_activation_function is Activations.SoftMax:
            self.loss = -np.sum(self.true_output * np.log(self.output + 1e-9))
        else:
            self.loss = np.sum(np.power(self.true_output - self.output, 2)) * (1/self.mini_batch_count)
        
        
        if self.output_activation_function is Activations.SoftMax:
            output_error = (self.output - self.true_output)
        else:
            output_error = (self.output - self.true_output) * self.output_activation_function(self.output, derivative=True)
        self.error.append(output_error)
        
        for layer in range(len(self.weights) - 1, -1, -1):
            
            if layer > 0:
                layer_input = self.hidden_layers[layer - 1]
            else:
                layer_input = self.inputs
                
            weight_gradient = self.error[-1] @ layer_input.T
            self.gradient.append(weight_gradient)
            
            if layer > 0:
                passed_error = self.weights[layer].T @ self.error[-1]
                next_node_error = passed_error * self.activation_function(self.hidden_layers[layer - 1], derivative=True)
                self.error.append(next_node_error)
            
        
    def updatePass(self):
        self.gradient.reverse()
        self.error.reverse()
        
        for index in range(len(self.weights)):
            self.weights[index] -= self.gradient[index] * self.learning_rate
            self.biases[index] -= self.error[index] * self.learning_rate

        self.reset()
    
    def reset(self):
        self.hidden_layers = []
        self.output = []
        self.error = []
    
    def lenDebug(self):
        print(f"Input Layer: {self.inputs.shape}")
        for i in range(len(self.hidden_layers)):
            print(f"Hidden Layer {i}: {self.hidden_layers[i].shape}")
        print(f"Output Layer: {self.output.shape}")
        
        print(f"\nInput Weight: {self.weights[0].shape}")
        for i in range(1, len(self.weights)):
            print(f"Hidden Weight {i}: {self.weights[i].shape}")
        print("")
            
    def arrDebug(self):
        print("-----Input-----")
        print(self.inputs)
        
        print("-----Output/Expected-----")
        print(self.output)
        print(self.true_output)
        
        print("-----Loss-----")
        print(self.loss)
        
        
if __name__ == "__main__":
    data = {
        "input": {
            "arr": [[1], [2]],
            "n_nodes": 2,
            "n_batch": 1
        },
        "hidden": {
            "n_layers": 2,
            "n_nodes": [5, 3]
        },
        "output": {
            "n_out": 1
        },
        "actual": [[math.sqrt(5)]],
        "activation": Activations.Sigmoid,
        "output_activation": Activations.RelU,
        "loss": "cross_entropy",
        "learning_rate": 0.1
    }
    
    nn = MultilayerPerceptron(data)
    for _ in range(100):
        nn.train()
        