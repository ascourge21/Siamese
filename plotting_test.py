import matplotlib
matplotlib.use('qt4agg')

import matplotlib.pyplot as plt
import numpy as np

a = np.random.rand(5, 5)
plt.imshow(a , interpolation='none', cmap='gray')
plt.show()