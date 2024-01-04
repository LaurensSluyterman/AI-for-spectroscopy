import matplotlib.pyplot as plt

def concentrations_plot(true_concentrations, predictions, labels, loc=None):
    for i in range(len(labels)):
        name = labels[i]
        plt.title(name)
        plt.xlabel('true concentration')
        plt.ylabel('predicted concentration')
        x = true_concentrations[:, i]
        y = predictions[:, i]
        plt.scatter(x, y)
        plt.axline((0, 0), slope=1, color="black", linestyle=(0, (5, 5)))

        # plt.close()
        # Determine the data range
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)

        # Calculate the required range to cover all data points
        combined_min = min(x_min, y_min)
        combined_max = max(x_max, y_max)
        plt.xlim(combined_min, combined_max)
        plt.ylim(combined_min, combined_max)
        plt.tight_layout()
        if loc:
            plt.savefig('./plots/predictedvsobserved' + name)
        else:
            plt.show()
