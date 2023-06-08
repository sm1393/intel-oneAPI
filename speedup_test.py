import numpy as np
from glob import glob
import matplotlib.pyplot as plt

def extractValuesFromFile(fileName):
    file_path = glob(fileName)
#     print(file_path)
    with open(file_path[0], 'r') as file:
        content = file.read()
        line = ""
        values = []
        for char in content:
            line = line + char
            if char == "\n":
                values.append(float(line[5:-1]))
                line = ""
        return values

def plot_speedup(title, without_optimization, with_optimization):
    """
    Plots a bar chart comparing the time taken by stock PyTorch model and the time taken by
    the model optimized by IPEX
    """
    data = {'without_optimization': without_optimization, 'with_optimization': with_optimization}
    model_type = list(data.keys())
    times = list(data.values())

    fig = plt.figure(figsize = (10, 5))

    # creating the bar plot
    plt.bar(model_type, times, color ='blue', width = 0.4)

    plt.title(title)
    plt.ylabel("Runtime (sec)")
    plt.title(f"Speedup acheived - {without_optimization/with_optimization:.2f}x")
    plt.savefig(title + '.png')

images_without_optimization = extractValuesFromFile("images_without_optimization.txt")
images_with_optimization = extractValuesFromFile("images_with_optimization.txt")
video_without_optimization = extractValuesFromFile("video_without_optimization.txt")
video_with_optimization = extractValuesFromFile("video_with_optimization.txt")

plot_speedup("Speed comparison for hundred images", images_without_optimization[-1], images_with_optimization[-1])

plot_speedup("Speed comparison for one minute video", video_without_optimization[-1], video_with_optimization[-1])