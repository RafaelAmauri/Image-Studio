import numpy as np

import include.dither as dither


def nearestColor(pixelColor: int, colorsAvailable: np.typing.ArrayLike) -> int:
    """
    Given a list of available colors, picks the one closest to pixelColor

    Args:
        pixelColor (int): The original color
        colorsAvailable (np.typing.ArrayLike): The list of available colors

    Returns:
        int: The color in colorsAvailable closest to pixelColor
    """
    candidate1Idx = np.searchsorted(colorsAvailable, pixelColor, "right") - 1
    candidate2Idx = candidate1Idx+1 if candidate1Idx < len(colorsAvailable)-1 else -1

    candidate1 = int(colorsAvailable[candidate1Idx])
    candidate2 = int(colorsAvailable[candidate2Idx])
    pixelColor = int(pixelColor)

    # The new pixel color is whichever of the two candidate colors are the closest to it
    quantizedColor = candidate1 if (pixelColor - candidate1) < (candidate2 - pixelColor) else candidate2

    return quantizedColor


def quantize(img: np.typing.ArrayLike, numberOfColors: int, useDithering=True) -> np.typing.ArrayLike:
    """Quantizes the image into an arbitrary number of colors.

    Args:
        img (np.typing.ArrayLike): The image array. Must be in the format (C, H, W)
        numberOfColors (int)     : How many colors are available. 
                                   The colors available are the interval [0, 255] 
                                   uniformly divided by this parameter.
        useDithering (bool)      : If True, uses dithering to help mitigate a narrow color palette

    Returns:
        np.typing.ArrayLike (np.uint8): The quantized image
    """
    # Sometimes, the dithering operation could cause an overflow if there's a pixel that's 
    # already at 255. If any non-zero dithering resudials get added to it, it would cause
    # a bit overflow. To be safe, we use np.int32 for just this one operation.
    img = np.copy(img).astype(np.int32)

    # Creates a uniformily spaced color distribution. It's a uniform division from 0 to 255, with numberOfColors elements.
    colorsAvailable = np.linspace(0, 255, numberOfColors, dtype=np.uint8)
    
    for channel in range(img.shape[0]):
        for row in range(img.shape[1]):
            for column in range(img.shape[2]):
                oldPixelColor = img[channel][row][column]

                # Quantize the pixel
                newPixelColor = nearestColor(oldPixelColor, colorsAvailable)
                img[channel][row][column] = newPixelColor

                if useDithering:
                    quantizationResidual = oldPixelColor - newPixelColor
                    residuals = dither.dither(quantizationResidual)
                    # Distribute the residuals. We clip the values so residuals are always in the [0, 255] range

                    if column + 1 < img.shape[2]:
                        # Update pixel to the right
                        img[channel][row][column+1]   = np.clip(img[channel][row][column+1]   + residuals[0], 0, 255)
                    
                    if row + 1 < img.shape[1]:
                        # Update pixel below
                        img[channel][row+1][column]   = np.clip(img[channel][row+1][column]   + residuals[2], 0, 255)
                        if column - 1 >= 0:
                            # Update pixel below and to the left
                            img[channel][row+1][column-1] = np.clip(img[channel][row+1][column-1] + residuals[1], 0, 255)
                        if column + 1 < img.shape[2]:
                            # Update pixel to the right
                            img[channel][row+1][column+1] = np.clip(img[channel][row+1][column+1] + residuals[3], 0, 255)

    
    img = img.astype(np.uint8)
    return img