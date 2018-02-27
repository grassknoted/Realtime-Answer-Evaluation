# Importing matplotlib for Visualizations
from matplotlib import pyplot
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pylab

# Importing operator to call operator.subtract
import operator


def plot_error(actualScores, predictedScores, graphName):

    # Create a figure window
    fig = plt.figure()

    # Assert that length of predicted and actual scores must be equal
    assert(len(predictedScores) == len(actualScores))

    # Set x-axis values to the Answer IDs
    x_axis_values = range(1, (len(predictedScores)+1) )

    # Calculate errors in predictions:
    predictionErrors = list(map(operator.sub, predictedScores, actualScores))

    #plt.scatter(x_axis_values, actualScores, color='red', linestyle = "None", label = 'Actual Scores', s = 0.3)
    #plt.scatter(x_axis_values, predictedScores, color='green', linestyle = "None", label = 'Predicted Scores', s = 0.3)

    # Plot errors:
    plt.errorbar(x_axis_values, predictedScores, yerr=predictionErrors, linestyle="None", color='red')

    # Location of the Legend
    pylab.legend(loc='upper left')

    # Graph title
    fig.suptitle(graphName)
    
    # X-axis Label
    plt.xlabel("Answer Number")

    # Y-axis Label
    plt.ylabel("Scores")

    # Display the graph
    plt.show()