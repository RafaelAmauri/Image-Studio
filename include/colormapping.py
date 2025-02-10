import numpy as np
import typing

def paletteConversion(img: np.typing.ArrayLike, LUT: typing.Dict) -> np.typing.ArrayLike:
    """Given an HSV image, this function converts the colors in the image according to a 
    color LUT https://en.wikipedia.org/wiki/3D_lookup_table

    Args:
        img (np.typing.ArrayLike): The HSV image
        LUT (typing.Dict)        : A color LUT. 
                                   Must be in the following format: { (lowerLimit, UpperLimit): (lowerTarget, upperTarget) }

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