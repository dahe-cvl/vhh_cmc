from matplotlib import pyplot as plt
import numpy as np


# method to visualize scatter plots
def visualizeScatterPlot(x, y, y_max, y_min):
    # plot angles over time
    plt.figure(1)
    plt.plot(np.arange(y), x)
    plt.ylim(ymax=y_max, ymin=y_min)
    plt.grid(True)
    plt.show()
    #plt.draw()
    #plt.pause(0.02)


# method to visualize hist plots


# method to visualize imageplots


# method to visualize configurable subplots


