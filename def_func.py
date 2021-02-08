import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image

# display image
def imshow(image):
  # numpy
  npimage = image.detach().numpy()
  plt.imshow(np.transpose(npimage, (1,2,0)))
  plt.show()


def save_batch(images, nrow, PATH):
  img = torchvision.utils.make_grid(images, nrow=nrow)
  img_out = np.transpose(img.detach().numpy().astype('float64'), (1,2,0))
  img_out = (255*img_out).astype('uint8')
  img_out = Image.fromarray(img_out)
  img_out.save(PATH)