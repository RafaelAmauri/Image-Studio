"""
Numpy doesn't support 2d convolution operations, so I implemented my own.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided


def convolve2d(img: np.typing.ArrayLike, kernel: np.typing.ArrayLike):
    """
    Performs a convolution operation (https://en.wikipedia.org/wiki/Convolution)
    in a 2d image. Given a kernel.

    Args:
        img (np.typing.ArrayLike): _description_
        kernel (np.typing.ArrayLike): _description_

    Returns:
        _type_: _description_
    """
    originalImgHeight, originalImgWidth = img.shape
    kernelHeight, kernelWidth           = kernel.shape

    # Determine the padding size (number of pixels to add on each side)
    # based on the kernel width. (Assumes an odd kernel size.)
    padding = kernelWidth // 2

    # Pad the image with padding on all sides.
    img = np.pad(img, (padding,padding))

    paddedImgHeight, paddedImgWidth = img.shape

    # The positions of the pixels for the each convolution. Starts at the top left corner pixel
    # and increments by 1 at every next convolution.
    pixelsFirstConvolve = np.asarray([np.arange(paddedImgWidth * rowId, paddedImgWidth * rowId + kernelWidth, dtype=np.uint32) for rowId in range(0, kernelWidth)])
    
    # Flatten the convolve positions, the padded img and the kernel
    pixelsFirstConvolve   = pixelsFirstConvolve.flatten()
    img                   = img.flatten()
    kernel                = kernel.flatten()

    # The amount of times we move the pixelsFirstConvolve matrix to the right is
    # the number of pixels in the flattened image minus the position that the last pixel in pixelsFirstConvolve points to.
    numIterations = len(img) - pixelsFirstConvolve[-1]

    # To make our code vectorized, instead of iteratively adding 1 to pixelsFirstConvolve in a for loop
    # we can precalculate a huge array with every number from 0 to numIterations and then add each of them to
    # pixelsFirstConvolve :)
    # This way, we'll have a huge array with every possible value of pixelsFirstConvolve. After doing this,
    # the code is easily vectorized.
    increments = np.arange(numIterations, dtype=np.uint32).reshape(-1, 1)

    # Some of the calculated increments result in the kernel centering on padded regions
    # (i.e., positions that don’t correspond to the original image). Here we compute the indices
    # of these invalid increments and remove them.
    fakeIncrements = np.stack([np.arange((idx * paddedImgWidth - padding * 2), idx * paddedImgWidth, dtype=np.uint32) for idx in range(1, originalImgHeight)]).flatten()
    
    # We delete the fake increments from increments
    increments = np.delete(increments, fakeIncrements).reshape(-1, 1)

    # And now the valid pixels are pixels in our huge array of valid positions. The valid positions are the pixelsFirstConvolve 
    # summed with every valid increment.
    validPixels   = img[pixelsFirstConvolve + increments]

    # Now we can finally do the convolution. Just do a dot product and then divide by the kernel size.
    img = np.dot(validPixels, kernel) / np.sum(kernel)

    # Reshape the convolved image into the original shape
    img = img.astype(np.uint8).reshape((originalImgHeight, originalImgWidth))

    return img
    