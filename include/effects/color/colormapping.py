import numpy as np
import typing


def changeColor(img: np.typing.NDArray, LUT: typing.Dict) -> np.typing.NDArray:
    """Given an HSV image, this function changes the colors within a specified hue with another hue!
    Use a color LUT as an argument and the function will change the pixels with a hue that fall within 
    a specific range with the other specified hue!

    Args:
        img (np.typing.NDArray): The HSV image
        LUT (typing.Dict)        : The color LUT. 
                                   Must be in the following format: { (originalHueLowerBound, originalHueUpperBound): (targetHueLowerBound, targetHueUpperBound) }

    Returns:
        np.typing.NDArray: The HSV image with converted colors.
    """
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            currentHue = img[row][column][0]
            for (originalHueLowerBound, originalHueUpperBound), (targetHueLowerBound, targetHueUpperBound) in LUT.items():
                if currentHue >= originalHueLowerBound and currentHue <= originalHueUpperBound:

                    newHue = targetHueLowerBound + (img[row][column][0] - originalHueLowerBound) * (targetHueUpperBound - targetHueLowerBound) / (originalHueUpperBound - originalHueLowerBound)
                    img[row][column][0] = newHue
    
    return img

    

def changeColorPaletteGrayscale(img: np.typing.NDArray, LUT: typing.Dict) -> np.typing.NDArray:
    """This function converts the color palette within an HSV image into a specified color palette!
    This works as a traditional color LUT (https://en.wikipedia.org/wiki/3D_lookup_table).
    The image should be black and white.

    Args:
        img (np.typing.NDArray): The HSV image
        LUT (typing.Dict): The color LUT. Must be in the following format: { originalColor: newColor }
        isRGB
    Returns:
        np.typing.NDArray: The HSV image with the new color palette!
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


def changeColorPaletteRGB(img: np.typing.NDArray, LUT: typing.Dict) -> np.typing.NDArray:
    """This function converts the color palette within an HSV image into a specified color palette!
    This works as a traditional color LUT (https://en.wikipedia.org/wiki/3D_lookup_table).
    The image should be black and white.

    Args:
        img (np.typing.NDArray) : The HSV image
        LUT (typing.Dict)         : The color LUT. Must be in the following format: { originalColor: newColor }
    Returns:
        np.typing.NDArray: The HSV image with the new color palette!
    """
    # The early implementation of this function was a mask that was applied for each pixel with a Hue
    # channel that matched each key in the LUT. This was slow for two reasons:
    
    # * There are too many keys! Small images without grayscale conversion usually have around 500 different hues, which is okay,
    # but images with large resolutions can get up to 2000 unique hues. This makes it really expensive to generate and apply a mask
    # for each of them.

    # * The values are originally in np.float32, which is fine, but the Vector Processors in modern CPUs tend to be limited in size,
    # so converting from np.float32 to a data type that uses less memory is a plus. Also, since operations on integers tend to be
    # much faster than operations on floating points, we can get a nice speedup if we convert from np.float32 to np.uint16

    # That is the code idea behind this function. We convert the Hue channel from the range [0, 360] to the range [0, 65535].
    # This give us 65535 different possible hues while using np.uint16, which is much faster for
    # vectorization than np.float16 or np.float32.
    
    # The max possible value for a uint16
    maxUint16 = 0xFFFF
    
    # And convert the Hue channel from [0, 360] to [0, maxUint16] in the image.
    hueChannelUint16 = (img[..., 0] / 360 * maxUint16).astype(np.uint16)

    # Separate the keys from the values in the LUT. Since we'll be looking up
    # for values in the [0, maxUint16] range, we have to convert the keys to that
    # range as well.
    LUTOriginalHues = np.asarray(list(LUT.keys()))
    LUTOriginalHues = (LUTOriginalHues / 360 * maxUint16).astype(np.uint16)

    LUTNewHSVValues = np.asarray(list(LUT.values()))


    # Create a huge array from 0 to maxUint16 that maps
    # every possible value for original Hues to the new values.
    hueHashMap = np.arange(maxUint16, dtype=np.float32)

    # Our hueHashMap is a huge array. For every key in the LUT, we modify the 
    # corresponding position that it maps to in the hueHashMap to store the new Hue.
    # Don't worry, the newHue is already in the [0, 360] range, so no conversion is needed.
    for originalHue, hsvValue in zip(LUTOriginalHues, LUTNewHSVValues):
        newHue = hsvValue[0]
        hueHashMap[originalHue] = newHue
    
    # The new hue channel is the old values from the hueChannelUint16
    # mapped (using our hashmap) to their new values.
    newHueChannel = hueHashMap[hueChannelUint16]

    # Now simply change the hue channel in the original image to the new hue
    # channel with the mapped values :)
    img[..., 0] = newHueChannel
    
    return img


def generatePalette(baseHue: int, availableColors: np.typing.NDArray, hueRange: int, isReversed: bool):
    """Given a initial Hue, it generates a new color palette with len(availableColors) different gradients of the hue parameter.
    Note that the hue in the palette is the same, the only change is in the saturation and brightness values of the given hue.

    Args:
        baseHue (int)                        : The hue in HSV format. Should be a value between 0 and 359.
        availableColors (np.typing.NDArray): The availableColors that were used for quantization.
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
    # The V component dictates the brightness value.
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