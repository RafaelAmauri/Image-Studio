import numpy as np


def floydSteinberg(quantizationResidual: int) -> np.typing.ArrayLike:
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


def orderedDithering(img, filterOption, availableColors):
    """TODO write later

    Args:
        img (_type_): _description_
        filterOption (_type_): _description_
        availableColors (_type_): _description_

    Returns:
        _type_: _description_
    """
    img = img.astype(np.float32)
    img = img / 255
    
    bayer2x2 = [
                [0, 2],
                [3, 1]
            ]
    
    bayer4x4 =  [
                    [ 0,  8,  2, 10],
                    [12,  4, 14,  6],
                    [ 3, 11,  1,  9],
                    [15,  7, 13,  5]
                ]
    
    bayer8x8 = [
                    [ 0, 32,  8, 40,  2, 34, 10, 42],
                    [48, 16, 56, 24, 50, 18, 58, 26],
                    [12, 44,  4, 36, 14, 46,  6, 38],
                    [60, 28, 52, 20, 62, 30, 54, 22],
                    [ 3, 35, 11, 43,  1, 33,  9, 41],
                    [51, 19, 59, 27, 49, 17, 57, 25],
                    [15, 47,  7, 39, 13, 45,  5, 37],
                    [63, 31, 55, 23, 61, 29, 53, 21]
                ]


    bayer2x2 = np.asarray(bayer2x2) / (len(bayer2x2) **2)
    bayer4x4 = np.asarray(bayer4x4) / (len(bayer4x4) **2)
    bayer8x8 = np.asarray(bayer8x8) / (len(bayer8x8) **2)

    precalculatedBayer = [bayer2x2, bayer4x4, bayer8x8][filterOption]

    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            xValue = row    % len(precalculatedBayer)
            yValue = column % len(precalculatedBayer)

            bayerValue = precalculatedBayer[xValue][yValue]
            
            for channel in range(img.shape[2]):
                originalGrayscale = img[row][column][channel]

                if len(availableColors) > 2:
                    adjustedGrayscale = originalGrayscale + bayerValue / len(availableColors)
                    
                    quantizedColorIdx = int(np.floor(adjustedGrayscale * len(availableColors)))
                    quantizedColorIdx = np.clip(quantizedColorIdx, 0, len(availableColors)-1)
                
                else:
                    if originalGrayscale > bayerValue:
                        quantizedColorIdx = 1
                    else:
                        quantizedColorIdx = 0
                  
                img[row][column][channel] = availableColors[quantizedColorIdx]

    img = img.astype(np.uint8)

    return img