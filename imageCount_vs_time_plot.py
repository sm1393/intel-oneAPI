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
    
without_optimization = extractValuesFromFile("images_without_optimization_intel_np.txt")

with_optimization = extractValuesFromFile("images_with_optimization_intel_np.txt")

plt.plot(without_optimization, color = "red", label = "without_optimization")
plt.plot(with_optimization, color = "green", label = "with_optimization")
plt.xlabel("No of images")
plt.ylabel("Time taken to infer")
plt.legend()

plt.savefig('image_plot_intel_np.png')