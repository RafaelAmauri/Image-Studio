# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as np

from cython.parallel import prange

cdef inline float clip(float value, float min, float max) nogil:
    if value <= min:
        return min
    if value >= max:
        return max
    
    return value


# Given a list of available colors, find the one that 'color' is the nearest to.
cdef inline float nearestColor(float color,
                               np.uint8_t *availableColors,
                               int availableColorsSize) nogil:

    # Binary search to find the nearest available
    cdef int low = 0
    cdef int high = availableColorsSize
    cdef int mid
    while low < high:
        mid = low + (high - low) // 2
        if availableColors[mid] < color:
            low = mid + 1
        else:
            high = mid

    if low == 0:
        return availableColors[0]
    if low == availableColorsSize:
        return availableColors[availableColorsSize - 1]

    cdef float dist1 = abs(color - availableColors[low - 1])
    cdef float dist2 = abs(color - availableColors[low])

    if dist1 < dist2:
        return availableColors[low - 1]
    else:
        return availableColors[low]


def floydSteinberg(np.ndarray[np.uint8_t, ndim=3] img, 
                   np.ndarray[np.uint8_t, ndim=1] availableColors):
    """
    Floyd-Steinberg Dithering unfortunately cannot be easily run in parallel because 
    distributing the quantization error has local dependencies with neighboring pixels :(

    For more details, this great paper by Quentin Guilloteau explains the problem quite well https://hal.science/hal-03594790/document.

    Fortunately, the channels are fully independent, so I can run Floyd-Steinberg for each channel in parallel and then
    stack them back together. This was essentially my strategy to run this in parallel. 
    
    This function uses a Floyd-Steinberg filter (https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering) to calculate 
    a dithering effect.

    The dithering works as follows:
        1. For each pixel in the image:
            2. We quantize the current pixel
            
            3. We compute the error of this pixel as the difference between the new value and the old value
            
            4. [Error Diffusion] We add a fraction of this error to the neighbouring pixels.
                For example, if the error for the current pixel is 42, we will add 7/16 x 42 to the value of the pixel on its
                right

    Args:
        img (np.typing.NDArray)             : The image array. Must be in the format (H, W, C)
        availableColors (np.typing.NDArray) : A list containing the colors available. Should start at 0 and 
                                                the last element should be 255.

    Returns:
        np.typing.NDArray (np.uint8): The quantized image
    """
    cdef int H = img.shape[0]
    cdef int W = img.shape[1]
    cdef int C = img.shape[2]

    # The predefined Floyd-Steinberg weights
    cdef float w0 = 7.0 / 16.0
    cdef float w1 = 3.0 / 16.0
    cdef float w2 = 5.0 / 16.0
    cdef float w3 = 1.0 / 16.0

    # Create a safety copy and convert to float32 because the quantization errors are often non-integer values.
    cdef np.ndarray[np.float32_t, ndim=3] out = np.copy(img).astype(np.float32)

    # I will not be using Python's Global Interpreter Lock (GIL) for the next part,
    # and not using the GIL requires declaring all the variables that will be
    # used in the calculation as C variables.
    cdef float originalColor, error
    cdef int row, column, channel

    # Convert from a numpy array to a C array + size
    cdef int availableColorsSize        = availableColors.shape[0]
    cdef np.uint8_t *availableColorsPtr = &availableColors[0]

    # Running all the channels in parallel in pure C requires not using the Python Global Interpreter Lock
    for channel in prange(C, nogil=True):
        for row in range(H):
            for column in range(W):
                originalColor = out[row, column, channel]

                # Quantize the pixel
                out[row, column, channel] = nearestColor(originalColor, availableColorsPtr, availableColorsSize)

                # Calculate the quantization error (difference between original color and new color)
                error = originalColor - out[row, column, channel]

                # Distribute the residuals. We clip the values so residuals are always in the [0, 255] range
                if column + 1 < W:
                    # Update pixel to the right
                    out[row, column+1, channel] = clip(out[row, column+1, channel] + error * w0, 0, 255)
                
                if row + 1 < H:
                    # Update pixel below
                    out[row+1, column, channel] = clip(out[row+1, column, channel] + error * w2, 0, 255)
                    
                    if column - 1 >= 0:
                        # Update pixel below and to the left
                        out[row+1, column-1, channel] = clip(out[row+1, column-1, channel] + error * w1, 0, 255)
                    if column + 1 < W:
                        # Update pixel to the right
                        out[row+1, column+1, channel] = clip(out[row+1, column+1, channel] + error * w3, 0, 255)
                
    return out.astype(np.uint8)
