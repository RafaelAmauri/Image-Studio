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

    

def changeColorPaletteGrayscale(img: np.typing.ArrayLike, LUT: typing.Dict) -> np.typing.ArrayLike:
    """This function converts the color palette within an HSV image into a specified color palette!
    This works as a traditional color LUT (https://en.wikipedia.org/wiki/3D_lookup_table).
    The image should be black and white.

    Args:
        img (np.typing.ArrayLike): The HSV image
        LUT (typing.Dict): The color LUT. Must be in the following format: { originalColor: newColor }
        isRGB
    Returns:
        np.typing.ArrayLike: The HSV image with the new color palette!
    """
    originalImgShape = img.shape

    # Convert the value channel to the range [0, 255]
    img = img * (1.0, 1.0, 255.0)
    img = img.astype(np.float32)
    
    # Reshape img so it's a 1-D array of individual pixels
    img = img.reshape(-1, img.shape[-1])

    for originalGrayscale, hsvValues in LUT.items():
        # For each gray value in the LUT, use a mask to find the pixels where the 
        # brightness value (V channel) is the same as the one in the color LUT
        maskOriginalGrayscale = img[..., 2] == originalGrayscale

        # And change them with the new HSV values in the color LUT!
        img[maskOriginalGrayscale] = hsvValues

    # Change the image back to the original shape
    img = img.reshape(originalImgShape)
    
    return img


def changeColorPaletteRGB(img: np.typing.ArrayLike, LUT: typing.Dict, hueChannelUint16) -> np.typing.ArrayLike:
    """This function converts the color palette within an HSV image into a specified color palette!
    This works as a traditional color LUT (https://en.wikipedia.org/wiki/3D_lookup_table).
    The image should be black and white.

    Args:
        img (np.typing.ArrayLike): The HSV image
        LUT (typing.Dict): The color LUT. Must be in the following format: { originalColor: newColor }
        isRGB
    Returns:
        np.typing.ArrayLike: The HSV image with the new color palette!
    """
    
    # Create a mapping for every possible Hue value
    originalHueHashMap = np.arange(65535, dtype=np.float32) 

    # And assign the correct value
    for originalHue, hsvValues in LUT.items():
        originalHueHashMap[originalHue] = hsvValues[0]
    
    hueChannelUint16 = originalHueHashMap[hueChannelUint16]

    img[..., 0] = hueChannelUint16
    
    return img


def generatePalette(baseHue: int, availableColors: np.typing.ArrayLike, hueRange: int, isReversed: bool):
    """Given a initial Hue, it generates a new color palette with len(availableColors) different gradients of the hue parameter.
    Note that the hue in the palette is the same, the only change is in the saturation and brightness values of the given hue.

    Args:
        baseHue (int)                        : The hue in HSV format. Should be a value between 0 and 359.
        availableColors (np.typing.ArrayLike): The availableColors that were used for quantization.
        hueRange (int)                       : By how much the hues in the palette can deviate from the baseHue.

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
    sComponent = smoothLinspace(0.2, 1.0, paletteSize, 1.15)
    # The V componen dictates the brightness value.
    vComponent = smoothLinspace(0.0, 1.0, paletteSize, 1.15)

    # Calculate the bounds for our hue values.
    hueLowerBound    = (baseHue - hueRange) % 360
    hueUpperBound    = (baseHue + hueRange) % 360

    # For example, if baseHue = 10 and hueRange = 20, we need to make sure that the hueComponent
    # is in the range [350, 30]. If we don't do this, it will go from [30, 350], the opposite of what
    # we want. We use hueOffset to fix this edge case.
    needsAdjusting   = False
    if hueLowerBound > hueUpperBound:
        needsAdjusting = True
        hueOffset = 360 - hueLowerBound
        hueLowerBound =  (hueLowerBound + hueOffset) % 360
        hueUpperBound =  (hueUpperBound + hueOffset) % 360

    # Now we can finally interpolate the values for our hues!
    hueComponent = smoothLinspace(hueLowerBound, hueUpperBound, paletteSize, 1)
    
    # In case the offset exists, we need to convert our hue values back into their original
    # range.
    if needsAdjusting:
        hueComponent    = hueComponent - hueOffset
        negativeHueMask = hueComponent < 0
        hueComponent[negativeHueMask] += 360

    # Create the color LUT
    colorLUT = dict()

    if isReversed:
        hueComponent = hueComponent[:: -1]

    # And finally add the values to it
    for idx in range(paletteSize):
        colorLUT[availableColors[idx]] = [hueComponent[idx], sComponent[idx], vComponent[idx]]

    
    return colorLUT