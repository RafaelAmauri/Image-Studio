import numpy as np

def dither(quantizationResidual: int) -> np.typing.ArrayLike:
    """Uses a Floyd-Steinberg filter (https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering) to calculate 
    a dithering effect. 

    In practice, it distributes the quantization residual in the following way:


    [0     S     7/16]


    [3/16  5/16  1/16]

    
    S is the current pixel that gave us the quantization residual. The pixel to the right of it gets 7/16 of the residual,
    so the pixel to the right is now:


    pixelToTheRight = pixelToTheRight + quantizationResidual*7/16.

    
    The same logic is applied to the other pixels.


    Args:
        quantizationResidual (int): The residual of the quantization operation for a given pixel

    Returns:
        np.typing.ArrayLike (np.float32): The quantizationResidual after being distributed according to the Floyd-Steinberg method
    """

    floydSteinbergWeights         = np.array([7/16, 3/16, 5/16, 1/16], dtype=np.float32)
    weightedQuantizationResiduals = quantizationResidual * floydSteinbergWeights
    

    return weightedQuantizationResiduals