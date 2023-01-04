import numpy as np
import matplotlib.pyplot as plt

scalograms = np.loadtxt("output_yt_split_1_obs_per_features.dat")
plt.plot(scalograms)

plt.show()