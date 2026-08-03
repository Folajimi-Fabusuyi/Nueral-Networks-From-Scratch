from utils.neural_network import NeuralNetwork
from utils.data_gen import pythagoreanGen

# quit()

pythagorean_in, pythagorean_out = pythagoreanGen(1000000)
# x_or_in = [[1, 1], [0, 1], [1, 0], [0, 0]]
# x_or_out = [[0], [1], [1], [0]]

# nn_settings = Settings(
#     input= pythagorean_in, 
#     output= pythagorean_out,
#     activation="leaky_relu",
#     output_activation="linear", 
#     mini_batch=64,
#     hidden="32",              
#     normalize=True,
#     # dropout_rate=0.35,
#     train_split=0.8,
#     lr=1e-3
# )

nn = NeuralNetwork("./utils/nn.mnet", pythagorean_in, pythagorean_out)

# Training loop
go_to_testing = ""
while go_to_testing != "y":
    nn.train_model(epoch=10, debug=True)
    go_to_testing = input("Quit[y/n]: ")


user_input = ""
while user_input.lower() not in ["q", "quit"]:
    user_input = input("Enter input for model prediction, separated by commas ['q' to quit]: ")
    if user_input in ["q", "quit"]:
        break
    data = [[float(number) for number in user_input.split(",")]]
    prediction = nn.predict(data)[0]
    print(round(prediction, 2))

nn.save_network("./utils/nn.mnet")

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