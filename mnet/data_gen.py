import math
from rich.progress import Progress
import random
import matplotlib.pyplot as plt
import numpy as np
import struct
from array import array
from os.path  import join

def pythagoreanGen(data_count=1000):

    with Progress() as progress:
        task = progress.add_task("[red]Generating...", total=((data_count // 3) * 2) // 4)
        task2 = progress.add_task("[blue]Edge cases...", total=int(data_count * 0.30))
        
        pythagorean_in = []
        pythagorean_out = []
        
        samples_per_bracket = ((data_count // 3) * 2) // 4
        brackets = [(0, 5), (5, 20), (20, 50), (50, 100)]
        
        for lower_bound, upper_bound in brackets:
            for _ in range(samples_per_bracket):
                # Generate floats instead of ints for smoother curves!
                i = random.uniform(lower_bound, upper_bound)
                j = random.uniform(lower_bound, upper_bound)
                
                k = math.sqrt(math.pow(i, 2) + math.pow(j, 2))
                
                pythagorean_in.append([i, j])
                pythagorean_out.append([k])
                progress.advance(task, 1)
                
        for _ in range(int(data_count * 0.15)):
            pythagorean_in.append([0.0, 0.0])
            pythagorean_out.append([0.0])
            progress.advance(task2, 1)

        for _ in range(int(data_count * 0.15)):
            i = random.uniform(0, 100)
            j = random.uniform(0, 100)
            
            k = math.sqrt(math.pow(i, 2) + math.pow(j, 2))
            
            pythagorean_in.append([i, j])
            pythagorean_out.append([k])
            progress.advance(task2, 1)
        
        indices = list(range(len(pythagorean_in)))
        random.shuffle(indices)
        
        pythagorean_in = [pythagorean_in[i] for i in indices]
        pythagorean_out = [pythagorean_out[i] for i in indices]

    return (pythagorean_in, pythagorean_out)

def sineGen(data_count=1000):
    with Progress() as progress:
        task = progress.add_task("[red]Generating...", total=data_count)
        
        sine_in = []
        sine_out = []
        
        samples_per_bracket = data_count // 4
        brackets = [(0, 90), (90, 180), (180, 270), (270, 360)]
        
        for lower_bound, upper_bound in brackets:
            for _ in range(samples_per_bracket):
                # Generate floats instead of ints for smoother curves!
                i = random.uniform(lower_bound, upper_bound)
                
                k = math.sin(i * math.pi / 180)
                
                sine_in.append([i])
                sine_out.append([k])
                progress.advance(task, 1)
                
        
        indices = list(range(len(sine_in)))
        random.shuffle(indices)
        
        sine_in = [sine_in[i] for i in indices]
        sine_out = [sine_out[i] for i in indices]

    return (sine_in, sine_out)

def imageArrayGen():

    class MnistDataloader(object):
        def __init__(self, training_images_filepath,training_labels_filepath,
                    test_images_filepath, test_labels_filepath):
            self.training_images_filepath = training_images_filepath
            self.training_labels_filepath = training_labels_filepath
            self.test_images_filepath = test_images_filepath
            self.test_labels_filepath = test_labels_filepath
        
        def read_images_labels(self, images_filepath, labels_filepath):        
            labels = []
            with open(labels_filepath, 'rb') as file:
                magic, size = struct.unpack(">II", file.read(8))
                if magic != 2049:
                    raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
                labels = array("B", file.read())        
            
            with open(images_filepath, 'rb') as file:
                magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
                if magic != 2051:
                    raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
                image_data = array("B", file.read())        
            images = []
            for i in range(size):
                images.append([0] * rows * cols)
            for i in range(size):
                img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
                img = img.reshape(28, 28)
                images[i][:] = img            
            
            return images, labels
                
        def load_data(self):
            x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
            x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
            return (x_train, y_train),(x_test, y_test) 

    with Progress() as progress:
        task1 = progress.add_task("[red]Extracting data from MNIST dataset...", total=1)
        task2 = progress.add_task("[red]Refactoring input data structure...", total=2)
        task3 = progress.add_task("[red]Refactoring input data types...", total=1)
        task4 = progress.add_task("[red]Refactoring output data types...", total=2)     
        task6 = progress.add_task("[red]Randomizing inputs...", total=1)
        task7 = progress.add_task("[red]Randomizing outputs...", total=1)

        # File Extraction
        input_path = './mnist/input'
        training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
        training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
        test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

        mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
        (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
        progress.advance(task1)

        # Input Refactoring
        inp = []
        for arr in [x_train, x_test]:
            for i in range(len(arr)):
                temp_list = []
                for j in range(len(arr[i])):
                    temp_list.extend(arr[i][j])
                inp.append(temp_list)
            progress.advance(task2)

        inp = list(np.array(inp).astype(np.float32))
        # for i in range(len(inp)):
        #     for j in range(len(inp[i])):
        #         inp[i][j] = int(inp[i][j])
        progress.advance(task3)

        # Output Refactoring    
        y_train = [[int(y)] for y in y_train]   
        progress.advance(task4)
        y_test = [[int(y)] for y in y_test]
        progress.advance(task4)

        task5 = progress.add_task("[red]Refactoring output data structure...", total=(len(y_train)+len(y_test)))
        out = []
        for arr in [y_train, y_test]:
            for i in arr:
                temp = [0] * 10
                temp[i[0]] = 1
                out.append(temp)
                progress.advance(task5)

        # Data Randomizer
        inp_index_list = list(range(len(inp)))
        random.shuffle(inp_index_list)
        inp = [inp[i] for i in inp_index_list]
        out = [out[i] for i in inp_index_list]
        progress.advance(task6)
        progress.advance(task7)

        return inp, out
    