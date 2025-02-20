"""
Numpy doesn't support 2d convolution operations, so I implemented my own.
"""

import numpy as np


def convolve2d(img: np.typing.ArrayLike, kernel: np.typing.ArrayLike) -> np.typing.ArrayLike:
    """
    Performs a convolution operation (https://en.wikipedia.org/wiki/Convolution) in a 2d image 
    using a given kernel.

    Note: Technically this function does a Cross-correlation (https://en.wikipedia.org/wiki/Cross-correlation)
    instead of a convolution, but since:
    (1) I only intend to implement symmetrical kernels like Box Blur and Gaussian Blur, it doesn't matter.
    (2) Some kernels that I intend to implement like Sobel actually don't do a convolution 
    but a cross-correlation

    then I don't see a reason to flip the kernel like it is formally required.

    Args:
        img (np.typing.ArrayLike): The image.
        kernel (np.typing.ArrayLike): The kernel. Must be odd-sized (3x3, 5x5, 7x7, etc). I also only tested this function with square kernels.

    Returns:
        np.typing.ArrayLike: The convolved image.
    """

    originalImgHeight, originalImgWidth = img.shape
    kernelHeight, kernelWidth           = kernel.shape

    # Determine the padding size (number of pixels to add on each side)
    # based on the kernel width. (Assumes an odd kernel size.)
    padding = kernelWidth // 2

    # Pad the image with padding on all sides.
    img = np.pad(img, ((padding, padding), (padding, padding)))

    # Creates a sliding window view into the array using the kernel shape.
    patches = np.lib.stride_tricks.sliding_window_view(img, kernel.shape)

    # Reshape the arrays so they match dimensionality and shape.
    patches = patches.reshape(-1, kernelHeight * kernelWidth)
    kernel  = kernel.flatten()

    # Some kernels (like Sobel) add up to zero when summing all the elements,
    # so doing np.sum(kernel) straight away could lead to a division by zero error.
    kernelSum = np.sum(kernel)
    kernelSum = kernelSum if kernelSum != 0 else 1

    # Perform the convolution operation.
    img = np.dot(patches, kernel) / kernelSum
    
    img = img.reshape(originalImgHeight, originalImgWidth).astype(np.uint8)
    
    return img