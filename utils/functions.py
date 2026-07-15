import numpy as np

class Loss:
    def CategoricalCrossEntropy(actual: np.ndarray, predicted: np.ndarray) -> int | float:
        return -np.sum(actual * np.log(predicted + 1e-9))
    
    # Going to be for sigmoid
    def BinaryCrossEntropy(actual: np.ndarray, predicted: np.ndarray, count: int) -> int | float:
        pass

    def MeanSquaredError(actual: np.ndarray, predicted: np.ndarray, count: int) -> int | float: 
        return np.sum(np.power(actual - predicted, 2)) * (1/count)
    
class Activations:
    # Output only activation
    def SoftMax(x: np.ndarray) -> np.ndarray:
        # Gives outputs as probabilities when paired with categorical cross entropy
        shifted_x = x - np.max(x)
        return np.exp(shifted_x) / np.sum(np.exp(shifted_x))
    
    # Either input or output activations
    def RelU(x: np.ndarray, derivative=False) -> np.ndarray:
        # Great for training for outputs without an upper bound but a strict lower bound at 0, usually not good for hidden layers
        if derivative == False:
            return np.maximum(0, x)
        return np.where(x > 0, 1, 0)
    
    def LeakyRelU(x: np.ndarray, derivative=False) -> np.ndarray:
        # Same as Relu but more leeway for lower bound and a viable hidden layers choice
        if derivative == False:
            return np.where(x > 0, x, 0.01 * x) 
        return np.where(x > 0, 1, 0.01)
    
    def Sigmoid(x: np.ndarray, derivative=False) -> np.ndarray:      
        # Great for predicting values between 0 and 1 when paired with binary cross entropy
        if derivative == False:
            return 1 / (1 + np.exp(-1 * x))
        return x * (1 - x)
    
    def Tanh(x: np.ndarray, derivative=False) -> np.ndarray:
        # Binds inputs/outputs to -1 and 1
        if derivative == False:  
            return np.tanh(x)
        return 1 - np.power(x, 2)
    
    def Linear(x: np.ndarray, derivative=False) -> np.ndarray:
        # Great for uncapped output values, negative or positive
        if derivative == False:
            return x
        return 1