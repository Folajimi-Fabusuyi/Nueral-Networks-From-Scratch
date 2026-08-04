from utils.neural_network import NeuralNetwork
from utils.func_collection import Activations
from utils.models import Settings
from utils.data_gen import pythagoreanGen, sineGen

def start_training(epochs=1000):
    # Training loop
    go_to_testing = ""
    while go_to_testing != "y":
        nn.train_model(epoch=epochs, debug=True)
        go_to_testing = input("Quit[y/n]: ")


    user_input = ""
    while user_input.lower() not in ["q", "quit"]:
        user_input = input("Enter input for model prediction, separated by commas ['q' to quit]: ")
        if user_input in ["q", "quit"]:
            break
        data = [[float(number) for number in user_input.split(",")]]
        prediction = nn.predict(data)[0]
        print(round(prediction, 4))


inp, out = ([[1, 1], [0, 1], [1, 0], [0, 0]], [[0], [1], [1], [0]])
# inp, out = pythagoreanGen(1000000)
# inp, out = sineGen(10000)

nn_settings = Settings(
    input= inp, 
    output= out,
    activation="leaky_relu",
    output_activation="relu", 
    # mini_batch=256,
    hidden="128, 32",              
    # normalize=True,
    # dropout_rate=0.35,
    train_split=1,
    lr=1e-3
)
filepath = "./utils/in.mnet"
out_filepath = "./utils/out.mnet"

start_training()
nn = NeuralNetwork(path=filepath, inp=inp, out=out)
nn = NeuralNetwork.create(settings=nn_settings)



nn.save_network(out_filepath)




# DONE: Add functionality for scaled input during prediction
# DONE: Add Dropout during forward pass
# DONE: Standardize input and output structure for passing into model
# DONE: Implement validation loss each epoch with train-test split
# DONE: Implement train_until conditional for model 
# DONE: Add Learning rate adapting in relation to validation_loss plateuing
# DONE: Implement model saving

# TODO: Implement dynamic model setting changes
# TODO: Implement rich terminal views
# TODO: Implement logging for networks and potential graphs
# TODO: Make tauri application using rust as backend, with embedded graph and dashboard