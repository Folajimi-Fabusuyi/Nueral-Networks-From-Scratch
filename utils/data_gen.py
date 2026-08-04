import math
from rich.progress import Progress
import random

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
    