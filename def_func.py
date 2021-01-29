import matplotlib.pyplot as plt
import numpy as np

# display image
def imshow(image):
#  image = image/2 + 0.5
  # numpy
  npimage = image.detach().numpy()
  plt.imshow(np.transpose(npimage, (1,2,0)))
  plt.show()
