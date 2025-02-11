import numpy as np

import include.dither as dither


def nearestColor(pixelColor: int, availableColors: np.typing.ArrayLike) -> int:
    """
    Given a list of available colors, picks the one closest to pixelColor

    Args:
        pixelColor (int): The original color
        availableColors (np.typing.ArrayLike): The list of available colors

    Returns:
        int: The color in availableColors closest to pixelColor
    """
    candidate1Idx = np.searchsorted(availableColors, pixelColor, "right") - 1
    candidate2Idx = candidate1Idx+1 if candidate1Idx < len(availableColors)-1 else -1

    candidate1 = int(availableColors[candidate1Idx])
    candidate2 = int(availableColors[candidate2Idx])
    pixelColor = int(pixelColor)

    # The new pixel color is whichever of the two candidate colors are the closest to it
    quantizedColor = candidate1 if (pixelColor - candidate1) < (candidate2 - pixelColor) else candidate2

    return quantizedColor


def quantize(img: np.typing.ArrayLike, availableColors: np.typing.ArrayLike, useDithering=True) -> np.typing.ArrayLike:
    """Quantizes the image into an arbitrary number of colors.

    Args:
        img (np.typing.ArrayLike)             : The image array. Must be in the format (H, W, C)
        availableColors (np.typing.ArrayLike) : A list containing the colors available. Should start at 0 and 
                                                the last element should be 255.
        useDithering (bool)                   : If True, uses dithering to help mitigate a narrow 
                                                color palette

    Returns:
        np.typing.ArrayLike (np.uint8): The quantized image
    """
    # Sometimes, the dithering operation could cause an overflow if there's a pixel that's 
    # already at 255. If any non-zero dithering resudials get added to it, it would cause
    # a bit overflow. To be safe, we use np.float32 for just this one operation.
    # At the end of the function, we convert the values back to np.uint8.
    img = np.copy(img).astype(np.float32)
    
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            for channel in range(img.shape[2]):
                oldPixelColor = img[row][column][channel]

                # Quantize the pixel
                newPixelColor = nearestColor(oldPixelColor, availableColors)
                img[row][column][channel] = newPixelColor

                if useDithering:
                    quantizationResidual = oldPixelColor - newPixelColor
                    residuals = dither.dither(quantizationResidual)
                    # Distribute the residuals. We clip the values so residuals are always in the [0, 255] range

                    if column + 1 < img.shape[1]:
                        # Update pixel to the right
                        img[row][column+1][channel]   = np.clip(img[row][column+1][channel]   + residuals[0], 0, 255)
                    
                    if row + 1 < img.shape[0]:
                        # Update pixel below
                        img[row+1][column][channel]   = np.clip(img[row+1][column][channel]   + residuals[2], 0, 255)
                        if column - 1 >= 0:
                            # Update pixel below and to the left
                            img[row+1][column-1][channel] = np.clip(img[row+1][column-1][channel] + residuals[1], 0, 255)
                        if column + 1 < img.shape[1]:
                            # Update pixel to the right
                            img[row+1][column+1][channel] = np.clip(img[row+1][column+1][channel] + residuals[3], 0, 255)

    
    img = img.astype(np.uint8)
    return img