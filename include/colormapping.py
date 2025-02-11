import numpy as np
import typing

def changeColor(img: np.typing.ArrayLike, LUT: typing.Dict) -> np.typing.ArrayLike:
    """Given an HSV image, this function changes the colors within a specified hue with another hue!
    Use a color LUT as an argument and the function will change the pixels with a hue that fall within 
    a specific range with the other specified hue!

    Args:
        img (np.typing.ArrayLike): The HSV image
        LUT (typing.Dict)        : The color LUT. 
                                   Must be in the following format: { (originalHueLowerBound, originalHueUpperBound): (targetHueLowerBound, targetHueUpperBound) }

    Returns:
        np.typing.ArrayLike: The HSV image with converted colors.
    """
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            currentHue = img[row][column][0]
            for (originalHueLowerBound, originalHueUpperBound), (targetHueLowerBound, targetHueUpperBound) in LUT.items():
                if currentHue >= originalHueLowerBound and currentHue <= originalHueUpperBound:

                    newHue = targetHueLowerBound + (img[row][column][0] - originalHueLowerBound) * (targetHueUpperBound - targetHueLowerBound) / (originalHueUpperBound - originalHueLowerBound)
                    img[row][column][0] = newHue
    
    return img


def changePalette(img: np.typing.ArrayLike, LUT: typing.Dict) -> np.typing.ArrayLike:
    """This function converts the color palette within an HSV image into a specified color palette!
    This works as a traditional color LUT (https://en.wikipedia.org/wiki/3D_lookup_table).
    The image should be black and white.

    Args:
        img (np.typing.ArrayLike): The HSV image
        LUT (typing.Dict): The color LUT. Must be in the following format: { originalColor: newColor }

    Returns:
        np.typing.ArrayLike: The HSV image with the new color palette!
    """

    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            currentGrayscale = img[row][column][2]
            if currentGrayscale in LUT:
                newHSV = LUT[currentGrayscale]
                img[row][column] = newHSV
    
    
    return img