import numpy as np

import include.effects.quantize as quantize


def distributeResiduals(quantizationResidual: np.typing.NDArray) -> np.typing.NDArray:
    """
    Returns a list for how the quantization residuals should be distributed.

    Args:
        quantizationResidual (int): The residual of the quantization operation for a given pixel

    Returns:
        np.typing.NDArray (np.float32): The quantizationResidual after being distributed according to the Floyd-Steinberg method
    """

    floydSteinbergWeights         = np.array([7/16, 3/16, 5/16, 1/16], dtype=np.float32)
    
    # Create new axis to broadcast. Output should be (nChannels, 4)
    weightedQuantizationResiduals = quantizationResidual[..., np.newaxis] * floydSteinbergWeights
    

    return weightedQuantizationResiduals


def floydSteinberg(img: np.typing.NDArray, availableColors: np.typing.NDArray) -> np.typing.NDArray:
    """
    WARNING: Floyd-Steinberg unfortunaly cannot run in parallel because calculating the quantization error depends 
    on the result of the previous iteration :(

    Fortunately, the channels are fully independent, so I can run floyd-steinberg for each channel in parallel and then
    stack them back together. This was essentially my strategy to run this in parallel.


    Uses a Floyd-Steinberg filter (https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering) to calculate 
    a dithering effect.

    In practice, it distributes the quantization residual in the following way:


    [0     S     7/16]


    [3/16  5/16  1/16]

    
    S is the current pixel that gave us the quantization residual. The pixel to the right of it gets 7/16 of the residual,
    so the pixel to the right is now:


    pixelToTheRight = pixelToTheRight + quantizationResidual*7/16.

    
    The same logic is applied to the other pixels.


    Args:
        img (np.typing.NDArray)             : The image array. Must be in the format (H, W, C)
        availableColors (np.typing.NDArray) : A list containing the colors available. Should start at 0 and 
                                                the last element should be 255.

    Returns:
        np.typing.NDArray (np.uint8): The quantized image
    """
    # Sometimes, the dithering operation could cause an overflow if there's a pixel that's 
    # already at 255. Since non-zero dithering resudials will get added to it, it would cause
    # a bit overflow. To be safe, we use np.float32 for just this one operation. Since the
    # dithering residuals are often floating points too, this was something we already had to anyway.
    # At the end of the function, we convert the values back to np.uint8.
    img = np.copy(img).astype(np.float32)
    
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            oldPixelColor = img[row][column].copy()
            
            # Quantize the pixel
            newPixelColor = quantize.nearestColor(oldPixelColor, availableColors)
            img[row][column] = newPixelColor
            
            # Calculate the residuals (difference between original color and new color)
            quantizationResidual = oldPixelColor - newPixelColor
            
            # Uses the Floyd-Steinberg algorithm to distribute the residuals.
            residuals = distributeResiduals(quantizationResidual)
            
            # Distribute the residuals. We clip the values so residuals are always in the [0, 255] range
            if column + 1 < img.shape[1]:
                # Update pixel to the right
                img[row][column+1]   = np.clip(img[row][column+1]   + residuals[..., 0], 0, 255)
            
            if row + 1 < img.shape[0]:
                # Update pixel below
                img[row+1][column]   = np.clip(img[row+1][column]   + residuals[..., 2], 0, 255)
                if column - 1 >= 0:
                    # Update pixel below and to the left
                    img[row+1][column-1] = np.clip(img[row+1][column-1] + residuals[..., 1], 0, 255)
                if column + 1 < img.shape[1]:
                    # Update pixel to the right
                    img[row+1][column+1] = np.clip(img[row+1][column+1] + residuals[..., 3], 0, 255)
                
    img = img.astype(np.uint8)

    return img


def orderedDithering(img: np.typing.NDArray, filterOption: int, availableColors: np.typing.NDArray):
    """
    Applies Ordered Dithering (https://en.wikipedia.org/wiki/Ordered_dithering) to the image. 

    Args:
        img (np.uint8)                       : The image
        filterOption (int)                   : The bayer kernel to use. 0 = 4x4 kernel, 1 = 8x8 kernel, 2 = 64x64 kernel.
        availableColors (np.typing.NDArray): The array of available colors.

    Returns:
        np.uint8: The dithered image
    """

    originalImgShape = img.shape

    img = img.astype(np.float32)
    img = img / 255

    # The precalculated threshold maps. These can theoretically be calculated on-the-fly, but it is much easier to just declare them like this.
    threshold2x2 = np.array([
                    [0.00, 0.50],
                    [0.75, 0.25]
                ])
    
    threshold4x4 =  np.array([
                    [0.0000, 0.5000, 0.1250, 0.6250],
                    [0.7500, 0.2500, 0.8750, 0.3750],
                    [0.1875, 0.6875, 0.0625, 0.5625],
                    [0.9375, 0.4375, 0.8125, 0.3125]
                ])


    threshold8x8 =  np.array([
                    [0.000000, 0.500000, 0.125000, 0.625000, 0.031250, 0.531250, 0.156250, 0.656250],
                    [0.750000, 0.250000, 0.875000, 0.375000, 0.781250, 0.281250, 0.906250, 0.406250],
                    [0.187500, 0.687500, 0.062500, 0.562500, 0.218750, 0.718750, 0.093750, 0.593750],
                    [0.937500, 0.437500, 0.812500, 0.312500, 0.968750, 0.468750, 0.843750, 0.343750],
                    [0.046875, 0.546875, 0.171875, 0.671875, 0.015625, 0.515625, 0.140625, 0.640625],
                    [0.796875, 0.296875, 0.921875, 0.421875, 0.765625, 0.265625, 0.890625, 0.390625],
                    [0.234375, 0.734375, 0.109375, 0.609375, 0.203125, 0.703125, 0.078125, 0.578125],
                    [0.984375, 0.484375, 0.859375, 0.359375, 0.953125, 0.453125, 0.828125, 0.328125]
                ])

    # Get the chosen bayer kernel
    thresholdMap = [threshold2x2, threshold4x4, threshold8x8][filterOption]

    # To vectorize with numpy, we repeat the bayer kernel in a tile pattern so it fully covers the image. This way, we can fully take advantage of vectorization to
    # speed up the code.
    thresholdMap = np.tile(thresholdMap, (img.shape[0] // len(thresholdMap) + 1, img.shape[1] // len(thresholdMap) + 1) )
    
    # Sometimes the tiling can make the bayer map larger than the image. To fix that, we simply "crop" the bayer map to the exact dimensions of the image.
    thresholdMap = thresholdMap[ : img.shape[0], : img.shape[1]]

    # And now we flatten both the image and the bayer map so they are both in the (H * W, C) format.
    img = img.reshape(-1, img.shape[-1])
    thresholdMap = thresholdMap.flatten()


    def ditherPixel(originalGrayscale, availableColors, thresholdMap):
        # The index of the new value for each pixel is their original value + the corresponding bayer matrix value in the X, Y coordinate / len(availableColors).
        # Since precalculatedBayer is the same dimension as the image channel, numpy vectorizes this section for us and it runs really fast!
        adjustedGrayscale = originalGrayscale + thresholdMap / len(availableColors)

        # Now we get the index of the color for the pixel's new grayscale value
        quantizedColorIdx = np.floor(adjustedGrayscale * len(availableColors))
        quantizedColorIdx = np.clip(quantizedColorIdx, 0, len(availableColors)-1).astype(np.uint8)

        # And assign it to that
        return availableColors[quantizedColorIdx]
    

    def ditherPixel2Colors(originalGrayscale, availableColors, thresholdMap):
        adjustedGrayscale = np.where(
                                originalGrayscale > thresholdMap,
                                availableColors[1],
                                availableColors[0]
        )

        return adjustedGrayscale

    if len(availableColors) > 2:
        img = np.stack([ditherPixel(img[:, channel]       , availableColors, thresholdMap) for channel in range(img.shape[-1])], axis=1)
    else:
        img = np.stack([ditherPixel2Colors(img[:, channel], availableColors, thresholdMap) for channel in range(img.shape[-1])], axis=1)

    
    # Now we simply return the image to its original shape.
    img = img.reshape(originalImgShape).astype(np.uint8)

    return img