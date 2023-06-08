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
    
without_optimization = extractValuesFromFile("video_without_optimization.txt")

with_optimization = extractValuesFromFile("video_with_optimization.txt")

xAxis = [20*i for i in range(len(with_optimization))]

plt.plot(xAxis, without_optimization, color = "red", label = "without_optimization")
plt.plot(xAxis, with_optimization, color = "green", label = "with_optimization")
plt.xlabel("No of frames")
plt.ylabel("Time taken to infer")
plt.legend()

plt.savefig('video_plot.png')