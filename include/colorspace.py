import numpy as np
from skimage.color import hsv2rgb as sk_hsv2rgb

def hsv2rgb(hsvImg: np.typing.ArrayLike) -> np.typing.ArrayLike:
    """Coming soon!

    Args:
        img (np.typing.ArrayLike): The HSV image.

    Returns:
        np.typing.ArrayLike: The RGB Image
    """

    hsvImg = hsvImg / (360, 1, 1)
    rgbImg = sk_hsv2rgb(hsvImg)
    rgbImg = rgbImg * 255
    
    rgbImg = np.asarray(rgbImg, dtype=np.uint8)

    return rgbImg


def rgb2hsv(img: np.typing.ArrayLike) -> np.typing.ArrayLike:
    """
    Converts an image from the RGB colorspace into the HSV colorspace (https://en.wikipedia.org/wiki/HSL_and_HSV)!

    WARNING: Lots of comments! I tried to make it as easy to understand as possible!

    Args:
        img (np.typing.ArrayLike): The RGB image.

    Returns:
        np.typing.ArrayLike: The HSV image.
    """
    # The specific formula for converting from RGB to HSV is in https://en.wikipedia.org/wiki/HSL_and_HSV#From_RGB
    # This is a vectorized version made with numpy.

    # Convert from np.uint8 to np.float32
    img = img.astype(np.float32)
    
    # Normalize the image
    img = img / 255.0
    originalImgShape = img.shape

    # Convert image to (H * W, C)
    img = img.reshape(-1, img.shape[-1])
    
    # Extract color channels
    redChannel   = img[..., 0]
    greenChannel = img[..., 1]
    blueChannel  = img[..., 2]

    # For every pixel, find out the largest and smallest value across the R, G and B channels
    Cmax = np.max(img, axis=1)
    Cmin = np.min(img, axis=1)
    
    # And create a mask for each channel. This mask tells us if a given pixels has its largest value in 
    # the Red, in the Green, or in the Blue channel. If there's two pixes and the first largest value in the Red channel and
    # the second is in the Blue channel for example, then maskRedChannel = [True, False], maskGreenChannel = [False, False] and
    # maskBlueChannel = [False, True].
    maxIdx = np.argmax(img, axis=1)
    maskRedChannel   = (maxIdx == 0)
    maskGreenChannel = (maxIdx == 1)
    maskBlueChannel  = (maxIdx == 2)

    # The difference between the largest and the smallest value across the R, G and B channels.
    delta = Cmax - Cmin

    # Where in the Cmax - Cmin matrix delta is not zero.
    nonZeroDelta = delta != 0

    # Create a hue and a value channel. We'll worry about the saturation channel later. The hue channel starts as all zeros, but 
    # it'll be assigned the right values later. The value Channel is just Cmax, so we don't have to do anything with it.
    # I just used hueChannel = np.zeros_like(redChannel) because the redChannel has the same number of pixels that our new hue channel will
    # have. It's just a quick and easy way to get an array with the right number of pixels.
    hueChannel        = np.zeros_like(redChannel)
    valueChannel      = Cmax

    # This next part is kinda complicated, but I'll try to explain the best way I can.
    # We have a brand new Hue channel that was just created. We have a separate variable for accessing the Red, Green and Blue channels directly (redChannel, greenChannel and blueChannel),
    # we have mask[Red,Green,Blue]Channel that shows us, for each pixel, if the max value came from the Red, the Green or the Blue channel. For example,
    # if we have two RGB pixel: [ [255, 0, 0], [0, 255, 0] ], then maskRedChannel = [True, False], maskGreenChannel = [False, True] and maskBlueChannel = [False, False].
    # In addition to that, we have a mask that shows us where delta wasn't zero, that is, the pixels where the R, G and B channels have different values.
    
    # When we do hueChannel[maskRedChannel   & nonZeroDelta], numpy accesses only the pixels where the red channel contained the highest value (maskRedChannel) and where delta wasn't zero (nonZeroDelta).
    # And numpy assigns **ONLY** to these pixels the value 60 * greenChannel[samePixel] - blueChannel[samePixel] / delta[samePixel] % 6.
    # It is super important to make sure that we use the same mask for the right-hand side of the equation because of that. This is why we're using the same mask on the left and right-hand sides.

    # The sequential code for this section is something like this:
    '''
        # If the largest value for this pixel comes from the Red channel    
        if CmaxIdx == 0:
            if delta != 0:
                hueValue = 60 * ((rgbValue[1] - rgbValue[2]) / delta % 6)

                if Cmax !=0:
                    saturationValue = delta / Cmax
                else:
                    saturationValue = 0
            else:
                hueValue = 0
                saturationValue = 0

        # If it comes from the Green channel
        elif CmaxIdx == 1:
            if delta != 0:
                hueValue = 60 * ((rgbValue[2] - rgbValue[0]) / delta + 2)
                
                if Cmax !=0:
                    saturationValue = delta / Cmax
                else:
                    saturationValue = 0
            else:
                hueValue = 0
                saturationValue = 0

        # Or if it comes from the Blue channel
        elif CmaxIdx == 2:
            if delta != 0:
                hueValue = 60 * ((rgbValue[0] - rgbValue[1]) / delta + 4)

                if Cmax !=0:
                    saturationValue = delta / Cmax
                else:
                    saturationValue = 0
            else:
                hueValue = 0
                saturationValue = 0
    '''

    # So, we're just replicating that, but in parallel :)
    # The masks are just a way for us to satisfy the conditions in the if statements before assigning a new value to the hue channel.
    hueChannel[maskRedChannel   & nonZeroDelta] = 60 * (((greenChannel[maskRedChannel  & nonZeroDelta] - blueChannel[maskRedChannel   & nonZeroDelta]) / delta[maskRedChannel   & nonZeroDelta]) % 6)
    hueChannel[maskGreenChannel & nonZeroDelta] = 60 * (((blueChannel[maskGreenChannel & nonZeroDelta] - redChannel[maskGreenChannel  & nonZeroDelta]) / delta[maskGreenChannel & nonZeroDelta]) + 2)
    hueChannel[maskBlueChannel  & nonZeroDelta] = 60 * (((redChannel[maskBlueChannel   & nonZeroDelta] - greenChannel[maskBlueChannel & nonZeroDelta]) / delta[maskBlueChannel  & nonZeroDelta]) + 4)

    # For the Saturation channel we do the same thing! Since there's one extra condition for assigining a new value to the saturation of each pixel,
    # all we have to do is add a new mask for where Cmax is non-zero and numpy will take care of it for us! We do it like so:
    saturationChannel = np.zeros_like(redChannel)
    nonZeroCmax = Cmax != 0
    saturationChannel[maskRedChannel   & nonZeroDelta & nonZeroCmax] = delta[maskRedChannel   & nonZeroDelta & nonZeroCmax] / Cmax[maskRedChannel   & nonZeroDelta & nonZeroCmax]
    saturationChannel[maskGreenChannel & nonZeroDelta & nonZeroCmax] = delta[maskGreenChannel & nonZeroDelta & nonZeroCmax] / Cmax[maskGreenChannel & nonZeroDelta & nonZeroCmax]
    saturationChannel[maskBlueChannel  & nonZeroDelta & nonZeroCmax] = delta[maskBlueChannel  & nonZeroDelta & nonZeroCmax] / Cmax[maskBlueChannel  & nonZeroDelta & nonZeroCmax]
    

    # Stack each of the Hue, Saturation and Value channels on top of one another.
    hsvImg = np.stack([hueChannel, saturationChannel, valueChannel], axis=1)

    # Right now our hsvImg array is a huge one-dimension array. We have to reshape it so it is in the [H, W, C] format.
    # Conviniently, originalImgShape is already in that format, so we just reuse that.
    hsvImg = np.reshape(hsvImg, originalImgShape)
    
    return hsvImg