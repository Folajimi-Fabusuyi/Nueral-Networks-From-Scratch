"""
Contains ModelConfig and NeuralNetwork classes
"""

import numpy as np
import random

from .registry import *
from .file_system import *
from .functions import *
from .models import BaseModel

class ModelConfig:
    '''
    Sets up config for neural network \n
    
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
        auto_normalize (bool): Determines if network should automatically scale down inputs and outputs to smaller values. Defaults to False.\n
        norm (tuple[float, float]): Sets the value inputs and ouputs are scaled down by, to this value. Takes precedence over auto_normalize\n
        dropout_rate (float): Percentage of random hidden layer nodes that are turned off during training forward passes. Defaults to 0.0.\n
    '''

    def __init__(self, inp: list[list[int]]=[], out: list[list[int]]=[], hidden="16, 8", 
                 mini_batch=0, lr=0.01, momentum=0.9, model_type="mp", 
                 activation="leaky_relu", output_activation="relu", auto_normalize=False,
                 norm: tuple[float, float]=(0.0, 0.0), dropout_rate=0.0, train_split=0.8):

        self._validateConfig(inp=inp, out=out, hidden=hidden,
                             mini_batch=mini_batch, lr=lr, momentum=momentum,
                             model_type=model_type, activation=activation,
                             output_activation=output_activation, auto_normalize=auto_normalize,
                             norm=norm, dropout_rate=dropout_rate, train_split=train_split)

        if self._isForTraining(inp):
            inp, out = np.array(inp), np.array(out)
            self.i_scaler, self.o_scaler = self._getScalers(inp, out, auto_normalize, norm)
            inp, out = self._normalize(inp, out)

            split_data = self._splitData(inp, out, train_split)
            self.train_input, self.train_output, self.validation_input, self.validation_output = split_data
            self.mini_batch = mini_batch if mini_batch != 0 else self.train_input.shape[1]

        else:
            self.i_scaler, self.o_scaler = norm[0], norm[1]
            self.train_input, self.train_output = np.array([]), np.array([])
            self.validation_input, self.validation_output = np.array([]), np.array([])
            self.mini_batch = mini_batch

        self.hidden = [int(layer) for layer in hidden.split(",")]

        self.dropout_rate = max(0.0, min(dropout_rate, 0.99))
        self.train_split = train_split
        self.momentum = momentum
        self.lr = lr
        self.auto_normalize = auto_normalize

        self.activation: Activation = ACTIVATION_DICT[activation]
        self.output_activation: Activation = ACTIVATION_DICT[output_activation]
        self.model_type: BaseModel = MODEL_DICT[model_type]
        self.batch = self._batchGen()

    @classmethod
    def fromFile(cls, data):
        '''Loads setting from saved mnet file'''

        return cls(
            hidden=data["settings"]["hidden"],
            mini_batch=data["settings"]["mini_batch"],
            lr=data["settings"]["learning_rate"],
            momentum=data["settings"]["momentum"],
            model_type=data["settings"]["model_type"],
            activation=data["settings"]["activation"],
            output_activation=data["settings"]["output_activation"],
            auto_normalize=data["settings"]["auto_normalize"],
            dropout_rate=data["settings"]["dropout_rate"],
            train_split=data["settings"]["train_split"],
            norm=(data["settings"]["i_scaler"], data["settings"]["o_scaler"])
        )


    def randomize(self):
        '''Randomizes train input and output indexes'''

        indices = list(range(self.train_input.shape[1]))  
        random.shuffle(indices)

        self.train_input = np.array([self.train_input.T[i] for i in indices]).T
        self.train_output = np.array([self.train_output.T[i] for i in indices]).T

    def _batchGen(self):
        '''Generator for batch inputs during training. Generator never stops iteration.'''

        index = 0

        while index < self.train_input.shape[1]:                     
            index += self.mini_batch
            inp = self.train_input[:, index - self.mini_batch: index]
            out = self.train_output[:, index - self.mini_batch: index]

            if index >= self.train_input.shape[1]:
                index = 0

            yield inp, out

    def _validationGen(self):
        '''Generator for batch inputs during validation. Stops iteration once exhausted'''

        index = 0

        while index < self.validation_input.shape[1]:
            index += self.mini_batch
            inp = self.validation_input.T[index - self.mini_batch: index].T
            out = self.validation_output.T[index - self.mini_batch: index].T

            yield inp, out

    def _update(self, **kwargs):
        '''
        Updates certain config attributes
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
                inp, out = np.array(value), np.array(kwargs.get("out"))
                norm = (self.i_scaler, self.o_scaler)
                self.i_scaler, self.o_scaler = self._getScalers(inp, out, self.auto_normalize, norm)
                inp, out = self._normalize(inp, out)
    
                split_data = self._splitData(inp, out, self.train_split)
                self.train_input, self.train_output, self.validation_input, self.validation_output = split_data

            elif key == "out": pass

            elif key == "lr": self.lr = value

            elif key == "momentum": self.momentum = value

            elif key == "mini_batch": 
                if self.mini_batch == 0:
                    self.mini_batch = value if value != 0 else self.train_input.shape[1]

            elif key in ["activation", "output_activation"]:
                if key == "activation": self.activation = ACTIVATION_DICT[value]
                self.output_activation = ACTIVATION_DICT[value]

            elif key == "norm":
                self.train_input = self.train_input * self.i_scaler
                self.train_output = self.train_output * self.o_scaler
                self.validation_input = self.validation_input * self.i_scaler
                self.validation_output = self.validation_output * self.o_scaler

                self.i_scaler, self.o_scaler = value

                self.train_input = self.train_input / self.i_scaler
                self.train_output = self.train_output / self.o_scaler
                self.validation_input = self.validation_input / self.i_scaler
                self.validation_output = self.validation_output / self.o_scaler

            elif key == "auto_normalize":
                if self.auto_normalize == False and value == False:
                    continue

                self.auto_normalize = value
                if self.auto_normalize:
                    self.train_input = self.train_input * self.i_scaler
                    self.train_output = self.train_output * self.o_scaler
                    self.validation_input = self.validation_input * self.i_scaler
                    self.validation_output = self.validation_output * self.o_scaler

                    self.i_scaler = max(self.train_input.max(), self.validation_input.max(), 
                                   (-1 * self.train_input).max(), (-1 * self.validation_input).max())
                    self.o_scaler = max(self.train_output.max(), self.validation_output.max(), 
                                   (-1 * self.train_output).max(), (-1 * self.validation_output).max())
                                
                    self.i_scaler = max(1e-2, float(self.i_scaler))
                    self.o_scaler = max(1e-2, float(self.o_scaler))

                    self.train_input = self.train_input / self.i_scaler
                    self.train_output = self.train_output / self.o_scaler
                    self.validation_input = self.validation_input / self.i_scaler
                    self.validation_output = self.validation_output / self.o_scaler
                elif (self.auto_normalize == False) and (self.i_scaler + self.o_scaler != 2):
                    self.train_input = self.train_input * self.i_scaler
                    self.train_output = self.train_output * self.o_scaler
                    self.validation_input = self.validation_input * self.i_scaler
                    self.validation_output = self.validation_output * self.o_scaler

                    self.i_scaler, self.o_scaler = 1, 1
            elif key == "train_split":
                print(self.train_input.shape[1], self.validation_input.shape[1])
                inp = np.concatenate((self.train_input, self.validation_input), axis=1).T
                out = np.concatenate((self.train_output, self.validation_output), axis=1).T

                split_data = self._splitData(inp, out, value)
                self.train_input, self.train_output, self.validation_input, self.validation_output = split_data


    def _validateConfig(self, **kwargs):
        '''Makes sure config values are without issues'''

        for key, value in kwargs.items():
            if (key == "inp"):
                if len(kwargs.get("out")) != len(value):
                    raise ConfigError(f"Input size {len(value)} != Output size {len(kwargs.get("output"))}")

            elif (key == "out"): pass

            elif (key == "hidden"):
                if type(value) is not str: 
                    raise ConfigError("Hidden needs to be in string fmt")
                for layer in value.split(","):
                    if not layer.replace(" ", "").isdigit(): 
                        raise ConfigError(f"Hidden layer {layer} is not an integer")
                    
            elif (key == "lr"):
                if value == 0: raise ConfigError(f"Model won't learn with {value} lr")

            elif (key == "momentum"):
                if value >= 1 or value <= 0: raise ConfigError(f"Momentum {value} not in range (0, 1)")

            elif (key == "mini_batch"):
                if value < 0: raise ConfigError("mini_batch has to be positive or 0")

            elif (key == "model_type"):
                if value not in REVERSE_MODEL_MAP.values():
                    raise ConfigError(f"Model type: ({value}) not implemented")

            elif (key in ["activation", "output_activation"]):
                if value not in REVERSE_ACTIVATION_MAP.values():
                    raise ConfigError(f"Activation function: ({value}) not implemented")

            elif (key == "norm"):
                if type(value) is not tuple and len(value) != 2:
                    raise ConfigError(f"Norm ({value}) is not formatted properly")

            elif (key == "auto_normalize"):
                if type(value) is not bool:
                    raise ConfigError("auto_normalize has to be a boolean")

            elif (key == "dropout_rate"):
                if value < 0 or value >= 1: raise ConfigError(f"Dropout not in range [0, 1)")

            elif (key == "train_split"):
                if value < 0 or value >= 1: raise ConfigError(f"Train_split not in range [0, 1)")

            else:
                raise ConfigError(f"{key} is an invalid config.")

    def _isForTraining(self, inp: list[list[int]] | list[None]) -> bool:
        return len(inp) != 0

    def _getScalers(self, inp: np.ndarray, out: np.ndarray, auto_normalize: bool, norm: tuple[float, float]) -> tuple[float | int, float | int]:
        '''Returns input and output scaling values'''

        if norm != (0.0, 0.0):
            i_scaler = norm[0]
            o_scaler = norm[1]
        elif auto_normalize:
            i_scaler = max((inp * -1).max(), inp.max())
            o_scaler = max((out * -1).max(), out.max())
                        
            i_scaler = max(1e-2, float(i_scaler))
            o_scaler = max(1e-2, float(o_scaler))
        else:
            i_scaler, o_scaler = (1, 1)

        return i_scaler, o_scaler

    def _normalize(self, inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''Normalizes input and output by scalers'''

        inp = inp / self.i_scaler
        out = out / self.o_scaler

        return inp, out

    def _splitData(self, inp: np.ndarray, out: np.ndarray, train_split: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''Splits given input into training and validation data'''

        train_data_end_index = int(inp.shape[0] * train_split)
        
        test_in = np.array(inp[:train_data_end_index]).T
        test_out = np.array(out[:train_data_end_index]).T

        validation_in = np.array(inp[train_data_end_index:]).T
        validation_out = np.array(out[train_data_end_index:]).T

        return (test_in, test_out, validation_in, validation_out)


class NeuralNetwork:
    '''
    Creates neural network from file or from given ModelConfig
    '''
    
    def __init__(self, config: ModelConfig):
        self.model: BaseModel = config.model_type(config)

    @classmethod
    def fromFile(cls, path:str):
        '''Creates neural network from file'''

        data = readMnet(path)
        config = ModelConfig.fromFile(data)

        nn = cls(config)
        nn.model.weights = data["weights"]
        nn.model.biases = data["biases"]

        return nn

    def train(self, epoch: int=1, debug=False):

        target_epoch = self.model.epoch + epoch

        while (self.model.epoch < target_epoch):
            self.model.train(debug)

    def predict(self, inp: list[list[int]]) -> list[int]:
        return self.model.predict(inp)

    def save(self, path: str):
        saveMnet(path, self)
