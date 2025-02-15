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
    originalImgShape = img.shape

    # Convert the value channel to the range [0, 255]
    img = img * (1.0, 1.0, 255.0)
    img = img.astype(np.float32)

    # Reshape img so it's a 1-D array of individual pixels
    img = img.reshape(-1, img.shape[-1])
    
    for key, hsvValues in LUT.items():
        # For each gray value in the LUT, use a mask to find the pixels where the 
        # brightness value (V channel) is the same as the one in the color LUT
        maskGrayscaleOriginal = img[..., 2] == key

        # And change them with the new HSV values in the color LUT!
        img[maskGrayscaleOriginal] = hsvValues

    # Change the image back to the original shape
    img = img.reshape(originalImgShape)

    return img


def generatePalette(baseHue: int, availableColors: np.typing.ArrayLike, paletteRange: int):
    """Given a initial Hue, it generates a new color palette with len(availableColors) different gradients of the hue parameter.
    Note that the hue in the palette is the same, the only change is in the saturation and brightness values of the given hue.

    Args:
        baseHue (int)                        : The hue in HSV format. Should be a value between 0 and 359.
        availableColors (np.typing.ArrayLike): The availableColors that were used for quantization.
        paletteRange (int)                   : By how much the hues in the palette can deviate from the baseHue.

    Returns:
        dict: A color LUT, where each value in the availableColors array is mapped to a HSV value.
    """
    
    # Instead of using np.linspace to create linearly spaced elements, I created this
    # function that smoothly interpolates between startValue and endValue with a 
    # specific exponent.
    # 
    # If the exponent is GREATER than 1, then SMALL VALUES GET MORE FREQUENT IN THE DISTRIBUTION.
    # If the exponent is LOWER   than 1, then LARGE VALUES GET MORE FREQUENT IN THE DISTRIBUTION.
    def smoothLinspace(startValue, endValue, numberElements, exponent):

        values = np.linspace(startValue, endValue, numberElements)
        
        # Normalize the values to get them in the [0, 1] range
        values = values / endValue

        # Raise them to the exponent
        values = values ** exponent

        # Get them back to their new values in the range [startValue, endValue]
        values = values * endValue

        return values


    paletteSize = len(availableColors)

    # The S component dictates the saturation value.
    sComponent = smoothLinspace(0.2, 0.8, paletteSize, 1.5)
    # The V componen dictates the brightness value.
    vComponent = smoothLinspace(0.0, 1.0, paletteSize, 1.15)

    # Calculate two analogous hues. They are in the range [baseHue - range, baseHue + range] 
    analogousHue1    = (baseHue + paletteRange) % 360
    analogousHue2    = (baseHue - paletteRange) % 360

    hueComponent = smoothLinspace(min(analogousHue1, analogousHue2), max(analogousHue1, analogousHue2), paletteSize, 1.15)

    colorLUT = dict()

    for idx in range(paletteSize):
        colorLUT[availableColors[idx]] = [hueComponent[idx], sComponent[idx], vComponent[idx]]

    return colorLUT