import numpy as np

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



def quantize(img: np.typing.ArrayLike, numberOfColors: int) -> np.typing.ArrayLike:
    """Quantizes the image into an arbitrary number of colors.

    Args:
        img (np.typing.ArrayLike): The image. Must be in the format (C, H, W)
        numberOfColors (int)     : How many colors are available. 
                                   The colors available are the interval [0, 255] 
                                   uniformly divided by this parameter.

    Returns:
        np.typing.ArrayLike: The quantized image
    """
    img = np.copy(img)
    colorsAvailable = np.linspace(0, 255, numberOfColors, dtype=np.uint8)

    for channel in range(img.shape[0]):
        for row in range(img.shape[1]):
            for pixel in range(img.shape[2]):
                img[channel][row][pixel] = nearestColor(img[channel][row][pixel], colorsAvailable)

    return img