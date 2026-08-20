"""
Collection of Loss and Activation functions
"""

import numpy as np
from .registry import *

class Loss:
    '''Collection of Loss Functions'''

    @staticmethod
    def CategoricalCrossEntropy(actual: np.ndarray, predicted: np.ndarray) -> int | float:
        return -np.mean(np.sum(actual * np.log(predicted + 1e-9), axis=0))
    
    # Going to be for sigmoid
    @staticmethod
    def BinaryCrossEntropy(actual: np.ndarray, predicted: np.ndarray, count: int) -> int | float:
        return NotImplementedError("Binary Cross Entropy for sigmoid not implemented")

    @staticmethod
    def MeanSquaredError(actual: np.ndarray, predicted: np.ndarray, count: int) -> int | float: 
        return np.sum(np.power(actual - predicted, 2)) * (1/count)


class Activation:
    '''Collection of Activation Functions'''
    
    # Output only activation
    @staticmethod
    def SoftMax(x: np.ndarray) -> np.ndarray:
        # Gives outputs as probabilities when paired with categorical cross entropy
        shifted_x = x - np.max(x, axis=0, keepdims=True)
        return np.exp(shifted_x) / np.sum(np.exp(shifted_x), axis=0, keepdims=True)
    
    # Either input or output activations
    @staticmethod
    def RelU(x: np.ndarray, derivative=False) -> np.ndarray:
        # Great for training for outputs without an upper bound but a strict lower bound at 0, usually not good for hidden layers
        if derivative == False:
            return np.maximum(0, x)
        return np.where(x > 0, 1, 0)

    @staticmethod
    def LeakyRelU(x: np.ndarray, derivative=False) -> np.ndarray:
        # Same as Relu but more leeway for lower bound and a viable hidden layers choice
        if derivative == False:
            return np.where(x > 0, x, 0.01 * x) 
        return np.where(x > 0, 1, 0.01)

    @staticmethod
    def Sigmoid(x: np.ndarray, derivative=False) -> np.ndarray:      
        # Great for predicting values between 0 and 1 when paired with binary cross entropy
        if derivative == False:
            return 1 / (1 + np.exp(-1 * x))
        return x * (1 - x)

    @staticmethod
    def Tanh(x: np.ndarray, derivative=False) -> np.ndarray:
        # Binds inputs/outputs to -1 and 1
        if derivative == False:  
            return np.tanh(x)
        return 1 - np.power(x, 2)

    @staticmethod
    def Linear(x: np.ndarray, derivative=False) -> np.ndarray | int:
        # Great for uncapped output values, negative or positive
        if derivative == False:
            return x
        return 1

# Registration

ACTIVATION_MAP[Activation.LeakyRelU] = 0
ACTIVATION_MAP[Activation.RelU] = 1
ACTIVATION_MAP[Activation.Sigmoid] = 2
ACTIVATION_MAP[Activation.SoftMax] = 3
ACTIVATION_MAP[Activation.Tanh] = 4
ACTIVATION_MAP[Activation.Linear] = 5

REVERSE_ACTIVATION_MAP[0] = "leaky_relu"
REVERSE_ACTIVATION_MAP[1] = "relu"
REVERSE_ACTIVATION_MAP[2] = "sigmoid"
REVERSE_ACTIVATION_MAP[3] = "softmax"
REVERSE_ACTIVATION_MAP[4] = "tanh"
REVERSE_ACTIVATION_MAP[5] = "linear"

ACTIVATION_DICT["leaky_relu"] = Activation.LeakyRelU
ACTIVATION_DICT["relu"] = Activation.RelU
ACTIVATION_DICT["sigmoid"] = Activation.Sigmoid
ACTIVATION_DICT["softmax"] = Activation.SoftMax
ACTIVATION_DICT["tanh"] = Activation.Tanh
ACTIVATION_DICT["linear"] = Activation.Linear

__all__ = [
    "Activation",
    "Loss"
]