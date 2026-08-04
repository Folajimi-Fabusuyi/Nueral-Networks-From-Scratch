import struct
import numpy as np
import hashlib
import rich.progress 

from utils.models import MultilayerPerceptron
from utils.func_collection import Activations
from utils.models import Settings

MODEL_MAP = {
    MultilayerPerceptron: 0
}

REVERSE_MODEL_MAP = {
    0: "mp"
}

ACTIVATION_MAP = {
    Activations.LeakyRelU: 0,
    Activations.RelU: 1,
    Activations.Sigmoid: 2,
    Activations.SoftMax: 3,
    Activations.Tanh: 4,
    Activations.Linear: 5
}

REVERSE_ACTIVATION_MAP = {
    0: "leaky_relu",
    1: "relu",
    2: "sigmoid",
    3: "softmax",
    4: "tanh",
    5: "linear"
}

HEADER_FMT = "<5sh"
HEADER_BYTES = 7

SETTINGS_FMT = "<4e2H2e4H"
SETTINGS_BYTES = 24

HIDDEN_LAYER_FMT = "<H"
HIDDEN_LAYER_BYTES = 2

WEIGHT_BIAS_HEADER_FMT = "<H"
WEIGHT_BIAS_HEADER_BYTES = 2

WEIGHT_SHAPE_FMT = "<HH"
WEIGHT_SHAPE_BYTES = 4

BIAS_SHAPE_FMT = "<H"
BIAS_SHAPE_BYTES = 2

WEIGHT_BIAS_FMT = "<f"
WEIGHT_BIAS_BYTES = 4

HASH_CHUNK_BYTES = 1024 * 1024
HASH_BYTES = 32

VERSION = 1

def readMnet(path: str) -> dict:
    '''Reads mnet files'''

    with rich.progress.open(path, "rb", description="[red]Reading...") as f:
        header_bin = f.read(HEADER_BYTES)
        settings_bin = f.read(SETTINGS_BYTES)

        metadata = struct.unpack(HEADER_FMT, header_bin)

        # General Settings extraction
        if metadata[0] == b"MyNet":
            settings_tuple = struct.unpack(SETTINGS_FMT, settings_bin)
            settings = {"learning_rate": settings_tuple[0],
                        "momentum": settings_tuple[1],
                        "dropout_rate": settings_tuple[2],
                        "train_split": settings_tuple[3],
                        "mini_batch": settings_tuple[4],
                        "normalize": settings_tuple[5],
                        "i_scaler": settings_tuple[6],
                        "o_scaler": settings_tuple[7],
                        "model_type": REVERSE_MODEL_MAP[settings_tuple[8]],
                        "activation": REVERSE_ACTIVATION_MAP[settings_tuple[9]],
                        "output_activation": REVERSE_ACTIVATION_MAP[settings_tuple[10]]}

            # Hidden Settings extraction
            layer_count = settings_tuple[11]
            hidden_layer_list = []
            for _ in range(layer_count):
                layer = struct.unpack(HIDDEN_LAYER_FMT, f.read(HIDDEN_LAYER_BYTES))[0]
                hidden_layer_list.append(str(layer))
            settings["hidden"] = ",".join(hidden_layer_list)

            # Confirming content is not corrupt
            previous_hash = f.read(HASH_BYTES)
            current_hash = hashlib.sha256()

            start_pos = f.tell()
            while True:
                chunk = f.read(HASH_CHUNK_BYTES)
                if not chunk:
                    break
                current_hash.update(chunk)
            current_hash = current_hash.digest()
            f.seek(start_pos)
            
            if current_hash != previous_hash:
                print("File Corrupt! Stopped reading")
                return 0

            # Weight Header extraction
            weight_count = struct.unpack(WEIGHT_BIAS_HEADER_FMT, f.read(WEIGHT_BIAS_HEADER_BYTES))[0]
            weight_sizes = []
            for _ in range(weight_count):
                weight_size = struct.unpack(WEIGHT_SHAPE_FMT, f.read(WEIGHT_SHAPE_BYTES))
                weight_sizes.append((weight_size[0], weight_size[1]))

            # Weight Extraction
            weights = []
            for size in weight_sizes:
                built_weight_array = []
                for _ in range(size[0]):
                    weight_array = []
                    for _ in range(size[1]):
                        unpacked_weight = struct.unpack(WEIGHT_BIAS_FMT, f.read(WEIGHT_BIAS_BYTES))[0]
                        weight_array.append(unpacked_weight)
                    built_weight_array.append(weight_array)
                weights.append(np.array(built_weight_array))

            # Bias Header Extraction
            bias_count = struct.unpack(WEIGHT_BIAS_HEADER_FMT, f.read(WEIGHT_BIAS_HEADER_BYTES))[0]
            bias_sizes = []
            for _ in range(bias_count):
                bias_size = struct.unpack(BIAS_SHAPE_FMT, f.read(BIAS_SHAPE_BYTES))[0]
                bias_sizes.append(bias_size)

            # Bias Extraction
            biases = []
            for size in bias_sizes:
                built_bias_array = []
                for _ in range(size):
                    b = f.read(WEIGHT_BIAS_BYTES)
                    unpacked_bias = struct.unpack(WEIGHT_BIAS_FMT, b)[0]
                    built_bias_array.append([unpacked_bias])
                biases.append(np.array(built_bias_array))

        else:
            print("Not of MNET file type")
            return 0
         
    return {"metadata": metadata, "settings": settings, "weights": weights, "biases": biases}

def saveMnet(path: str, network):
    '''
    Saves network dictionary to mnet file.
    Settings are inserted in the binary in this order
        (filetype, version, lr, momentum, dropout, train_split, mini_batch, normalize, i_scaler, o_scaler, 
         model_type, activation, out_activation, hidden_layers_count, hidden_layer_nodes,
         hash, weights, biases)
    '''

    with rich.progress.Progress() as progress:
        task = progress.add_task("[red]Converting to Binary...", total=6)

        settings = network.settings
        model = network.model

        progress.update(task, description="[red]Converting Metadata")
        metadata_b = getHeaderBinary()
        progress.advance(task)

        progress.update(task, description="[red]Converting Settings...")
        settings_b = getSettingsBinary(settings)
        progress.advance(task)

        progress.update(task, description="[orange]Converting Weights...")
        weigth_b = getWeightsBinary(model)
        progress.advance(task)

        progress.update(task, description="[yellow]Converting Biases...")
        bias_b = getBiasesBinary(model)
        progress.advance(task)

        progress.update(task, description="[yellow]Creating Hash...")
        hash = hashlib.sha256()

        for start_index in range(0, len(weigth_b), HASH_CHUNK_BYTES):
            hash.update(weigth_b[start_index: start_index + HASH_CHUNK_BYTES])

        for start_index in range(0, len(bias_b), HASH_CHUNK_BYTES):
            hash.update(bias_b[start_index: start_index + HASH_CHUNK_BYTES])
        hash = hash.digest()
        progress.advance(task)

        progress.update(task, description="[green]Saving File...")
        with open(path, "wb") as f:
            f.write(metadata_b)
            f.write(settings_b)
            f.write(hash)
            f.write(weigth_b)
            f.write(bias_b)
        progress.advance(task)


# Sub functions for saving
def getHeaderBinary() -> bytes:
    header_binary = struct.pack(HEADER_FMT, b"MyNet", VERSION)
    return header_binary

def getSettingsBinary(settings: Settings) -> bytes:
    learning_rate = settings.lr
    momentum = settings.momentum
    dropout = settings.dropout_rate
    train_split = settings.train_split
    mini_batch = settings.mini_batch
    normalize = settings.normalize
    i_scaler = settings.i_scaler
    o_scaler = settings.o_scaler

    model_type = MODEL_MAP[settings.model_type]
    activation = ACTIVATION_MAP[settings.activation]
    out_activation = ACTIVATION_MAP[settings.output_activation]

    hidden_layers_count = 0
    hidden_layer_nodes = []
    for layer in settings.hidden:
        hidden_layer_nodes.append(layer)
        hidden_layers_count += 1

    settings_binary = struct.pack(
        SETTINGS_FMT, learning_rate, momentum,
        dropout, train_split, mini_batch,
        normalize, i_scaler, o_scaler, model_type, activation, out_activation,
        hidden_layers_count,
    )

    for layer in hidden_layer_nodes:
        settings_binary += struct.pack(HIDDEN_LAYER_FMT, layer)

    return settings_binary

def getWeightsBinary(model) -> bytes:
    weight_count = len(model.weights)
    matrix_sizes = []
    for weight in model.weights:
        matrix_sizes.append(weight.shape)

    weights_binary = struct.pack(WEIGHT_BIAS_HEADER_FMT, weight_count)
    for size in matrix_sizes:
        weights_binary += struct.pack(WEIGHT_SHAPE_FMT, size[0], size[1])

    for weight_array in model.weights:
        for node_weights in weight_array: 
            for weight in node_weights:
                weights_binary += struct.pack(WEIGHT_BIAS_FMT, weight)


    return weights_binary

def getBiasesBinary(model) -> bytes:
    bias_count = len(model.biases)
    bias_sizes = [biases.shape[0] for biases in model.biases]

    biases_binary = struct.pack(WEIGHT_BIAS_HEADER_FMT, bias_count)
    for size in bias_sizes: 
        biases_binary += struct.pack(WEIGHT_BIAS_HEADER_FMT, size)

    for bias_array in model.biases:
        for bias in bias_array:
            biases_binary += struct.pack(WEIGHT_BIAS_FMT, bias[0])

    return biases_binary