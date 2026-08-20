"""
Collection of Model Types
"""

import numpy as np
from abc import ABC, abstractmethod

from .registry import *
from .functions import *
from .core import *


class BaseModel(ABC):
    '''Abstract base class for all neural networks'''

    @abstractmethod
    def train(self): 
        '''Trains model for one epoch'''   
        pass

    @abstractmethod
    def predict(self, inp): 
        '''Uses existing weights and biases to predict input'''
        pass

    @abstractmethod
    def updateConfig(self, **kwargs): pass

    @abstractmethod
    def _importConfig(self, config): 
        '''Loads config into model'''
        pass


class MultilayerPerceptron(BaseModel):
    '''Network consisting of forwardpass and backpropagation through layers.'''

    def __init__(self, config: ModelConfig):
        self.epoch = 0
        self.epoch_ended = False

        self.raw_hidden_layers: list[np.ndarray] = []
        self.hidden_layers: list[np.ndarray] = []
        
        self.raw_output: np.ndarray = []
        self.output: np.ndarray = []
        
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        
        self.velocity: list[np.ndarray] = []

        self.error: list[np.ndarray] = []
        self.gradient: list[np.ndarray] = []
        
        self.batch_losses: list[int] = []
        self.train_loss = 0
        self.validation_loss = 0
        
        self.plateau = 0
        self.long_plateau = 0
        self.best_validation_loss = np.inf

        self.config = config
        self._importConfig(config)
        self._initialize()

    def train(self, debug=False) -> None:
        inp, true_out = next(self.config.batch)
        out = self._forwardPass(inp)
        self._backPropagation(inp, out, true_out)
        self._updatePass()

        if debug:
            # os.system('cls' if os.name == 'nt' else 'clear')
            print("\033[H\033[J", end="")
            print(f"Epoch {self.epoch}")
            print(f"Training_Loss: {round(self.train_loss, 10)}") 
            print(f"Validation_Loss: {round(self.validation_loss, 10)}")
            print(f"Batch: {len(self.batch_losses)}/{self.config.train_input.shape[1]//self.mini_batch}")
            if self.dropout_rate:
                print(f"Best Val_Loss: {round(self.best_validation_loss, 10)}")
                print()
                print(f"Lr: {round(self.learning_rate, 10)}")
                print(f"Plateu: {self.plateau}")
                print(f"Long_Plateu: {self.long_plateau}")

    def predict(self, inp: list[list[int]]) -> list[int]:        
        inp = np.array(inp).T / self.config.i_scaler # Normalizes input to match size of training input
        out = self._forwardPass(inp, predicting=True)
        out = (out * self.config.o_scaler).flatten().tolist()
        
        self._reset()
        return out

    def updateConfig(self, **kwargs):        
        '''
        Updates config of given arguments

        Valid: 
            inp
            out
            lr
            momentum
            mini_batch
            activation
            output_activation
            norm
            auto_normalize
            dropout_rate
            train_split
        '''
        for key, value in kwargs.items():
            if key == "inp":
                input = value
                output = kwargs.get("out")

                if not (input and output): 
                    raise UpdateError("Input and Output need to be provided")
                elif len(input) != len(output):
                    raise UpdateError(f"Input length ({len(input)}) != Output Length ({len(output)})")
                elif len(input[0]) != self.weights[0].shape[1]:
                    raise UpdateError(f"Shape error from inputs: {len(input[0])} should be {self.weights[0].shape[1]}")

                self.config._update(inp=input, out=output)

            elif key == "out": pass

            elif key == "lr": 
                if value <= 0:
                    raise UpdateError(f"Lr {value} has to be positive")
                self.config._update(lr=value)

            elif key == "momentum":
                if value >= 1 or value <= 0: 
                    raise UpdateError(f"Momentum {value} not in range (0, 1)")
                self.config._update(momentum=value)

            elif key == "mini_batch":
                if value < 0:
                    raise UpdateError("Mini_batch has to be positive or 0")
                self.config._update(mini_batch=value)

            elif key in ["activation", "output_activation"]:
                if value not in REVERSE_ACTIVATION_MAP.values():
                    raise UpdateError(f"Activation function: ({value}) not implemented")
                if key == "activation": self.config._update(activation=value)
                else: self.config._update(output_activation=value)

            elif key == "norm":
                if type(value) is not tuple and len(value) != 2:
                    raise UpdateError(f"Norm ({value}) is not formatted properly")
                self.config._update(norm=value)

            elif key == "auto_normalize":
                if type(value) is not bool: 
                    raise UpdateError("auto_normalize has to be a boolean")
                self.config._update(auto_normalize=value)

            elif key == "dropout_rate":
                if value < 0 or value >= 1: raise UpdateError(f"Dropout not in range [0, 1)")
                self.config._update(dropout_rate=value)

            elif key == "train_split":
                if value < 0 or value >= 1: raise UpdateError(f"Train_split not in range [0, 1)")
                self.config._update(train_split=value)

            else:
                raise UpdateError(f"{key} is not a valid config to change")

        self._importConfig(self.config)

    def _forwardPass(self, inp, predicting=False) -> np.ndarray:
        '''Aggregates hidden and output layers'''

        self.actual_batch = inp.shape[1]

        # Hidden layer propagation
        for index in range(self.layer_count):                         
            if index == 0:
                raw_hidden_layer = self.weights[index] @ inp + self.biases[index]
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
        output = self.output_activation(raw_output)

        return output

    def _backPropagation(self, inp: np.nd_array, out, true_out): 
        '''Finds gradients and error to update weights and biases'''
                
        true_output = true_out
        output = out
        
        # This calculation here is really only for us to view how the model is doing
        # Might possibly be useful for more advanced learing rate adjustments
        if self.output_activation is Activation.SoftMax:
            self.batch_losses.append(Loss.CategoricalCrossEntropy(true_output, output))
        else:
            self.batch_losses.append(Loss.MeanSquaredError(true_output, output, self.actual_batch))
        
        # The derivative of the loss
        if self.output_activation in [Activation.SoftMax, Activation.Linear]:
            output_error = (output - true_output)
        else:
            output_error = (output - true_output) * self.output_activation(output, derivative=True)
        self.error.append(output_error)
        
        # Gradient and error propagation
        for layer in range(len(self.weights) - 1, -1, -1):           
            if layer > 0:
                layer_input = self.hidden_layers[layer - 1]
            else:
                layer_input = inp

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
        if self.batch_index < self.config.train_input.shape[1]:
            self.batch_index += self.mini_batch
            self.epoch_ended = False
        else: self.epoch_ended = True 

    def _updatePass(self) -> None:
        '''Updates weights and biases by gradient'''

        clip_value = 1.0
        self.gradient = [np.clip(gradient, -clip_value, clip_value) for gradient in self.gradient]
        self.gradient.reverse()
        self.error.reverse()
        
        # Momentum implementation
        if self.velocity == []:
            self.velocity = [np.zeros(self.weights[i].shape) for i in range(len(self.weights))]
        self.velocity = [(self.momentum * self.velocity[i]) - (self.learning_rate * self.gradient[i]) for i in range(len(self.weights))]

        # Weight and bias nudges
        for index in range(len(self.weights)):
            # self.weights[index] -= self.gradient[index] * self.learning_rate
            self.weights[index] += self.velocity[index]
            self.biases[index] -= np.sum(self.error[index], axis=1, keepdims=True) * self.learning_rate
        
        
        # This means that an epoch has ended
        if self.epoch_ended: self._epochEnds()
        self._reset()

    def _validate(self) -> None: 
        '''Tests model on untrained data'''
                
        batch_losses = []
        validation_batch = self.config._validationGen()
        
        for inp, true_out in validation_batch:
            predicted_output = self._forwardPass(inp, predicting=True)

            if self.output_activation is Activation.SoftMax: batch_loss = Loss.CategoricalCrossEntropy(true_out, predicted_output)
            else: batch_loss = Loss.MeanSquaredError(true_out, predicted_output, inp.shape[1])
            
            batch_losses.append(batch_loss)
            self._reset()
        
        self.validation_loss = np.mean(batch_losses)
        self._reset()

    def _epochEnds(self) -> None:
        # Increment epoch count, reset batch index
        self.epoch += 1
        self.epoch_ended = True
        self.batch_index = self.mini_batch
        
        # Recalculate average loss and reset loss log for current epoch
        self.train_loss = np.mean(self.batch_losses)
        self.batch_losses = []
        
        # Valdidate epoch with untrained data
        self._reset()
        self._validate()
        self.config.randomize()
        
        # Loss Scheduler - Reduce learning_rate on plateu
        if self.dropout_rate:
            if self.validation_loss >= self.best_validation_loss:
                self.plateau += 1
            else:
                self.best_validation_loss = self.validation_loss
                self.plateau = 0
                self.long_plateau = 0
                
            if self.plateau >= 25:
                reduced_learning_rate = (self.learning_rate * 0.5)
                self.learning_rate = max(reduced_learning_rate, 1e-6)
                self.plateau = 0
                
                if self.learning_rate <= 1e-6:
                    self.long_plateau += 1
            
            # Revive learning_rate if plateau has gone on for too long
            if self.long_plateau >= 5:
                if self.validation_loss > 3e5: self.learning_rate *= 1000
                else: self.learning_rate *= 100
                self.long_plateau = 0

    def _importConfig(self, config) -> None: 
        # Node counts
        self.input_node_count = config.train_input.shape[0] if config.train_input.shape else 0
        self.nodes_per_layer = config.hidden
        self.layer_count = len(self.nodes_per_layer)
        self.output_node_count = config.train_output.shape[0] if config.train_output.shape else 0
        
        # Batch variables
        self.mini_batch = config.mini_batch
        self.batch_index = self.mini_batch
        self.actual_batch = 0
        
        self.dropout_rate = config.dropout_rate
        self.dropout_masks: list[np.ndarray] = []
        
        # Activation functions
        self.activation: Activation = config.activation
        self.output_activation: Activation = config.output_activation

        # Backpropagation variables
        self.learning_rate = config.lr
        
        self.momentum = config.momentum

    def _initialize(self) -> None: 
        '''Aggregates weights and biases'''    
                   
        # Weight aggregation
        self.weights.append(np.random.randn(self.nodes_per_layer[0], self.input_node_count) * 0.01) # First layer weights
        for layer in range(self.layer_count):
            if layer < self.layer_count - 1:
                self.weights.append(np.random.randn(self.nodes_per_layer[layer+1], self.nodes_per_layer[layer]) * 0.01)
                continue
            self.weights.append(np.random.randn(self.output_node_count, self.nodes_per_layer[layer]) * 0.01)
            
        # Hidden layers' randomized biases are lowered to reduce inital network noise
        for layer in range(self.layer_count):
            self.biases.append(np.random.randn(self.nodes_per_layer[layer], 1) * 0.01)
        
        # Output biases are set to 0 to reduce initial network noise as well
        self.biases.append(np.zeros((self.output_node_count, 1)))

    def _reset(self) -> None:
        self.hidden_layers = []
        self.output = []
        self.error = []
        self.gradient = []
        self.dropout_masks = []
        self.predict_input = None


# Registry
MODEL_MAP[MultilayerPerceptron] = 0

REVERSE_MODEL_MAP[0] = "mp"

MODEL_DICT["mp"] = MultilayerPerceptron

__all__ = [
    "BaseModel",
    "MultilayerPerceptron"
]