from mnet.core import *
from mnet.data_gen import pythagoreanGen, imageArrayGen, sineGen
import math
import random
from rich import print

def pythagorean_test():
    inp, out = pythagoreanGen()
    config = ModelConfig(inp=inp, out=out, mini_batch=64, activation="leaky_relu", 
                        output_activation="linear", auto_normalize=True, hidden="32, 8")
    nn = NeuralNetwork(config)

    nn.train(1000, debug=True)

    test_values = [(random.randint(0, 10), random.randint(0, 10)) for _ in range(20)]
    test_values.append((0, 0))

    for a, b in test_values:
        prediction = round(nn.predict([[a, b]])[0], 2)
        actual = round(math.sqrt(math.pow(a, 2) + math.pow(b, 2)), 2)

        print(f"[{a}, {b}]   Predicted: {prediction} | Actual: {actual}")
        
    # nn.save("./test.mnet")

def image_test():
    def start_training(epochs=10, test_type=0):
        # Training loop
        # go_to_testing = ""
        # while go_to_testing != "y":
        #     nn.train_model(epoch=epochs, debug=True)
        #     go_to_testing = input("Quit[y/n]: ")
        nn.train(epoch=epochs, debug=True)

        if test_type == 0:
            user_input = ""
            while user_input.lower() not in ["q", "quit"]:
                user_input = input("Enter input for model prediction, separated by commas ['q' to quit]: ")
                if user_input in ["q", "quit"]:
                    break
                data = [[float(number) for number in user_input.split(",")]]
                prediction = nn.predict(data)[0]
                print(round(prediction, 4))
        else:
            inputs = 1000
            counter = 0
            start = random.randint(0, len(out) - inputs)
            for i in range(start, start + inputs):
                pred = nn.predict([inp[i]])
                if int(pred.index(max(pred))) == out[i].index(max(out[i])):
                    counter += 1
                # print(f"[red]Prediction: {int(pred.index(max(pred)))}", f"[green]Answer: {out[i].index(max(out[i]))}")
            print(f"Results: {(counter/inputs) * 100}%")

            return (counter/inputs) * 100


    inp, out = imageArrayGen()
    filepath = "./models/img.mnet"
    nn = NeuralNetwork.fromFile(path=filepath)
    values = []
    for i in range(100):
        values.append(start_training(0, test_type=1))
    arr = np.array(values)
    print(f"Average Accuracy: {arr.mean()}")
    print(f"Max Accuracy: {arr.max()}")
    print(f"Min Accuracy: {arr.min()}")

pythagorean_test()


# DONE: Add functionality for scaled input during prediction
# DONE: Add Dropout during forward pass
# DONE: Standardize input and output structure for passing into model
# DONE: Implement validation loss each epoch with train-test split
# DONE: Implement train_until conditional for model 
# DONE: Add Learning rate adapting in relation to validation_loss plateuing
# DONE: Implement model saving
# DONE: Implement input shuffling every epoch
# DONE: Implement dynamic model config changes

# IN_PROGRESS: Implement rich terminal views

# TODO: Implement logging for networks and potential graphs
# TODO: Make tauri application using rust as backend, with embedded graph and dashboard