import numpy as np
import json

def sigmoid(x) -> np.array:
    return 1 / (1 + np.exp(-1 * x))

def sigmoid_derivative(x) -> np.array:
    return sigmoid(x) * (1 - sigmoid(x))

def round_values(x, n=4):
    return round(x, n)

class XOR_MultilayerPerceptron:
           
    def train(self, training_input, expected_output, lr, epochs=0):
        self.x = training_input
        self.y = expected_output
        self.lr = lr # Learning rate of nueral network
        self.epochs = epochs
        self.until_epoch = True if self.epochs else False
        
        self.n_in = self.x.shape[0]
        self.batch = self.x.shape[1]
        self.n_hidden = 3
        self.n_out = 1
        
        left_bound, right_bound = -1, 1  # To make our initial biases allow negative numbers when randomizing
        
        self.W1 = np.random.rand(self.n_hidden, self.n_in)
        self.b1 = np.random.random_sample((self.n_hidden, 1)) * (right_bound - left_bound) + left_bound

        self.W2 = np.random.rand(self.n_out, self.n_hidden)
        self.b2 = np.random.random_sample((self.n_out, 1)) * (right_bound - left_bound) + left_bound        
        
        self.forward_pass()
        self.backpropagate()
        
        self.current_epoch = 1
        self.logs = {"loss": {}}
        
        with open("results/result.txt", "w") as f:
            if self.until_epoch:
                while self.current_epoch <= self.epochs:
                    self.log_to_files(f)
                    
                    self.current_epoch += 1
                    self.forward_pass()
                    self.backpropagate()
            else:
                while self.L > .01:
                        self.log_to_files(f)

                        self.current_epoch += 1
                        self.forward_pass()
                        self.backpropagate() 
                        
        json.dump(self.logs, open("results/result.json", "w"), indent=True)
                
        
    def forward_pass(self):          
        self.pre_h = self.W1 @ self.x + self.b1
        self.h = sigmoid(self.pre_h)
        
        self.pre_y_hat = self.W2 @ self.h + self.b2
        self.y_hat = sigmoid(self.pre_y_hat)
    
    def backpropagate(self):        
        self.L = np.mean((self.y_hat - self.y)**2) # Loss function
        
        # Start by finding derivatives of output with relation to L
        # Only going to have a single output so we dont have to worry about derivating mean
        dy_hat = (2/self.batch) * (self.y_hat - self.y)
        dpre_y_hat = dy_hat * sigmoid_derivative(self.pre_y_hat) # The multiplication in the activation derivation step is because we are multiplying element-wise instead of dot product
        
        # Important values from layer 2
        dW2 = dpre_y_hat @ self.h.transpose()
        db2 = np.sum(dpre_y_hat, axis=1, keepdims=True)
        
        ## Starting layer 1 derivation, these are intermediate values needed to find weights and bias derivatives
        dh = self.W2.transpose() @ dpre_y_hat
        dpre_h = dh * sigmoid_derivative(self.pre_h)
        
        # Important values from layer 1
        dW1 = dpre_h @ self.x.transpose()
        db1 = np.sum(dpre_h, axis=1, keepdims=True)
        
        # This is where the values are nudged by their 'gradients'?
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
    def log_to_files(self, f):
        self.logs["loss"][self.current_epoch] = round_values(self.L, 9)
        # self.logs["output"][self.current_epoch] = [i for i in np.vectorize(round_values)(self.y_hat)[0]]
        
        print(f"Epoch {self.current_epoch} | Loss: {round_values(self.L, 9)} | Output: {np.vectorize(round_values)(self.y_hat)}\n")
        f.write(f"Epoch {self.current_epoch} | Loss: {round_values(self.L, 9)} | Output: {np.vectorize(round_values)(self.y_hat)}\n")
                    
    
    

neural_network = XOR_MultilayerPerceptron()

input = np.array([[0, 1, 1, 0],
                  [1, 0, 1, 0]])
output = np.array([[1, 1, 0, 0]])

neural_network.train(input, output, lr=0.1, epochs=20000)


# Make a loss curve graph for MSE + sigmoid
# Make a loss curve graph for MSE + relU

# Make a loss curve graph for Cross-entropy + sigmoid
# Make a loss curve graph for Cross-entropy + relU