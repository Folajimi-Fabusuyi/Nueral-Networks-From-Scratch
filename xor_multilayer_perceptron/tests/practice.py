import numpy as np
import random
import threading

def failed_forward_pass():
    # Practicing forwardpass exclusively

    input_layer_neurons = 5
    hidden_layer_neurons = 3
    output_layer_neurons = 2

    input_layer = np.array([round(random.random(), 1) for _ in range(input_layer_neurons)])
    hidden_layer = np.zeros(hidden_layer_neurons)
    output_layer = np.zeros(output_layer_neurons)

    input_layer_weights = np.array([round(random.random(), 1) for _ in range(input_layer_neurons)])
    hidden_layer_weights = np.array([round(random.random(), 1) for _ in range(hidden_layer_neurons)])

    hidden_layer_biases = np.zeros(hidden_layer_neurons)
    output_layer_biases = np.zeros(output_layer_neurons)
    for layer in [hidden_layer_biases, output_layer_biases]:
        for i in range(layer.shape[0]):
            bias = random.random()
                
            if random.random() >= 0.5:
                bias *= -1
                
            layer[i] = bias

    for i in range(hidden_layer_neurons):
        weighted_sum = input_layer.dot(input_layer_weights) + hidden_layer_biases[i]
        activation = max(0, weighted_sum) #RelU activation function
            
        hidden_layer[i] = round(activation, 1)
        
    for i in range(output_layer_neurons):
        weighted_sum = hidden_layer.dot(hidden_layer_weights) + output_layer_biases[i]
        activation = max(0, weighted_sum) #RelU activation function
            
        output_layer[i] = round(activation, 1)


    def debug():
        print("----------Layer Activations------------")
        print(input_layer)
        print(hidden_layer)
        print(output_layer)
        print()

        print("----------  Layer Weights  ------------")
        print(input_layer_weights)
        print(hidden_layer_weights)
        print()

        print("----------  Layer  Biases  ------------")
        print(hidden_layer_biases)
        print(output_layer_biases)

    debug()
    
    
def forward_pass():
    def relU(n):
        return np.maximum(0, n)
    
    n_input = 3
    n_hidden = 5
    n_out = 2 
    
    left_bound, right_bound = -1, 1
    
    input = np.random.rand(n_input, 1)
    
    W1 = np.random.rand(n_hidden, n_input)
    
    b1 = np.random.random_sample((n_hidden, 1)) * (right_bound - left_bound) + left_bound
    hidden = relU((W1 @ input) + b1)
    
    W2 = np.random.rand(n_out, n_hidden)
    b2 = np.random.random_sample((n_out, 1)) * (right_bound - left_bound) + left_bound
    output = relU((W2 @ hidden) + b2)
    
    return output

def backpropagate(y, x, W2, W1, b1, b2, lr):
    
    y_ = np.array([[10]], [10])
    loss = np.mean((y_ - y)**2)
    
    grad_W1 = (y_ - y) * W2 * x * -2 * lr
    grad_W2 = (y_ - y) * (W1 * x + b1) * -2 * lr
    grad_b1 = (y_ - y) * W2 * -2 * lr
    grad_b2 = (y_ - y) * -2 * lr
    
    W2 -= grad_W2
    W1 -= grad_W1
    b2 -= grad_b2
    b1 -= grad_b1
    
    return (W1, W2, b1, b2)
    
output = forward_pass()
backpropagate(output, 0.001)