import numpy as np
import matplotlib.pyplot as plt

plt.axis([0, 10, 0, 1])

for i in range(10):
    y = np.random.random()
    plt.plot(i, y, color='green', marker='o', linestyle='dashed',
    linewidth=2, markersize=1)
    plt.pause(0.05)

plt.show()