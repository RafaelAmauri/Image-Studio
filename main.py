import numpy as np
import PIL.Image

import include.quantize as quantize

enableGrayscale = False

# Open the image
img = PIL.Image.open("./assets/dog.jpg")

# Convert to grayscale if so desired
if enableGrayscale:
    img = img.convert("L")

img = np.asarray(img)

# To make the code work with grayscale, we need to add a fake "channel" dimension. This makes 
# it easier to interact with the rest of the code
if enableGrayscale:
    img = np.expand_dims(img, axis=0)

# Convert from [H, W, C] into the [C, H, W] format
img = img.transpose(2, 0, 1)


img = quantize.quantize(img, 2)


# Convert from the [C, H, W] format into [H, W, C]
img = img.transpose(1, 2, 0)


# Remove the fake "channel" dimension
if enableGrayscale:
    img = img.squeeze(0)


# Save the image
img = PIL.Image.fromarray(img)
img.save("./processed.png")