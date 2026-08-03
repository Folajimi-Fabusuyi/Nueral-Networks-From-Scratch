from typing import override

from utils.models import Settings
from utils.file_system import readMnet, saveMnet


class NeuralNetwork:
    '''
    Creates a neural network based off of created settings or imports settings from mnet file\n
    
    Parameters:\n
        settings (Settings): The settings of the nueral network\n

        path (str): Path of saved neural network\n
        inp (list[list[int]]): Input\n
        out (list[list[int]]): Output
    '''

    def __init__(self, settings: Settings):        
        self.settings = settings
        self.model = settings.model_type(settings)

    @override
    def __init__(self, path: str, inp: list[list[int]], out: list[list[int]]):
        data = readMnet(path)
        self.settings = Settings(
            inp, out,
            hidden=data["settings"]["hidden"],
            mini_batch=data["settings"]["mini_batch"],
            lr=data["settings"]["learning_rate"],
            momentum=data["settings"]["momentum"],
            model_type=data["settings"]["model_type"],
            activation=data["settings"]["activation"],
            output_activation=data["settings"]["output_activation"],
            normalize=data["settings"]["normalize"],
            dropout_rate=data["settings"]["dropout_rate"],
            train_split=data["settings"]["train_split"]
        )

        self.model = self.settings.model_type(self.settings, True)
        self.model.weights = data["weights"]
        self.model.biases = data["biases"]
        
    def train_model(self, epoch=1, debug=False):
        '''Trains model for number of epochs'''
        target_epoch = self.model.epoch + epoch
        
        while self.model.epoch != target_epoch:
            self.model.train(debug)
        
    def predict(self, input: list[list[int]]) -> list[int]:
        '''Uses existing weights and biases to predict input'''
        return self.model.predict(input)

    def save_network(self, path: str):
        saveMnet(path, self)