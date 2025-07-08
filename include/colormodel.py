"""
Colormodel.py is where I implement my own color model conversions. For more clarification of what even is
a color model, check https://en.wikipedia.org/wiki/Color_model
"""

import numpy as np


def rgb2grayscale(img: np.typing.NDArray) -> np.typing.NDArray:
    """Converts an image from RGB to Grayscale. 
    I am following this formula https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_rgb_to_gray.html
    which states that the conversion weights for each channel should be 
    
    Y = 0.2125 R + 0.7154 G + 0.0721 B

    because of how human vision works.

    Args:
        img (np.typing.NDArray): The RGB image

    Returns:
        np.typing.NDArray: Tge grayscale image
    """
    img     = img.astype(np.float64)

    weights = np.array([0.2125, 0.7154, 0.0721])
    
    # This is pretty straightforward, its just a weighted sum.
    img     = img * weights
    img     = np.sum(img, axis=2).astype(np.uint8)
    
    # Add a fake 'channel' dimension. This makes it easier to make grayscale images interact with the rest of the code.
    img = np.expand_dims(img, axis=2)

    return img



def rgb2hsv(img: np.typing.NDArray) -> np.typing.NDArray:
    """
    Converts an image from the RGB color model into the HSV color model (https://en.wikipedia.org/wiki/HSL_and_HSV#From_RGB)!

    WARNING: Lots of comments! I tried to make it as easy to understand as possible!

    Args:
        img (np.typing.NDArray): The RGB image.

    Returns:
        np.typing.NDArray: The HSV image.
    """
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



def hsv2rgb(hsvImg: np.typing.NDArray) -> np.typing.NDArray:
    """
    The formula for conversion can be found in https://en.wikipedia.org/wiki/HSL_and_HSV#HSV_to_RGB

    
    Args:
        hsvImg (np.typing.NDArray): The HSV image.

    Returns:
        np.typing.NDArray: The RGB Image
    """
    # Save the original image dimensions and reshape the array
    originalShape = hsvImg.shape
    hsvImg = hsvImg.reshape(-1, hsvImg.shape[-1])

    # Separate each of the channels in the HSV image
    hueChannel        = hsvImg[..., 0]
    saturationChannel = hsvImg[..., 1]
    valueChannel      = hsvImg[..., 2]

    # And divide the Hues by 60.
    hueChannel        = hueChannel / 60

    # The chroma is just the intensity difference between the darkest and brightest color in the RGB 
    # space at a given “slice” of hue.
    chromaComponent   = valueChannel * saturationChannel

    # X is essentially how far along that slice we are as the hue moves between the 
    # “corners” of the color hexagon (hence the mod 2).
    x = chromaComponent * (1 - np.abs((hueChannel % 2 - 1)))
    
    # Start the image as all zeros.
    rgbImg = np.zeros_like(hsvImg)

    # Here we have to find a mask for when the value in the hue channel (that we divided by 60)
    # is between these pairs.
    mask0h1 = (hueChannel >= 0) & (hueChannel < 1)
    mask1h2 = (hueChannel >= 1) & (hueChannel < 2)
    mask2h3 = (hueChannel >= 2) & (hueChannel < 3)
    mask3h4 = (hueChannel >= 3) & (hueChannel < 4)
    mask4h5 = (hueChannel >= 4) & (hueChannel < 5)
    mask5h6 = (hueChannel >= 5) & (hueChannel < 6)
    
    # Now, for each mask, we reassign the values like so. We unfortunately can't do this more efficiently because not only the mask changes, but also the order
    # of each new component. Sometimes it's Chroma, X, 0, other times it's X, Chroma, 0, etc.
    rgbImg[mask0h1] = np.stack([chromaComponent[mask0h1],           x[mask0h1],                         np.zeros_like(hueChannel[mask0h1])], axis=-1)
    rgbImg[mask1h2] = np.stack([x[mask1h2],                         chromaComponent[mask1h2],           np.zeros_like(hueChannel[mask1h2])], axis=-1)
    rgbImg[mask2h3] = np.stack([np.zeros_like(hueChannel[mask2h3]), chromaComponent[mask2h3],           x[mask2h3]],                         axis=-1)
    rgbImg[mask3h4] = np.stack([np.zeros_like(hueChannel[mask3h4]), x[mask3h4],                         chromaComponent[mask3h4]],           axis=-1)
    rgbImg[mask4h5] = np.stack([x[mask4h5],                         np.zeros_like(hueChannel[mask4h5]), chromaComponent[mask4h5]],           axis=-1)
    rgbImg[mask5h6] = np.stack([chromaComponent[mask5h6],           np.zeros_like(hueChannel[mask5h6]), x[mask5h6]],                         axis=-1)


    # 'm' is an offset that lifts the [0..chroma] range to [m..(m+chroma)], so that the final R, G and B channels each end up in [0..1] after adding m.
    m = valueChannel - chromaComponent

    # Here is where we sum 'm' to each channel    
    rgbImg = rgbImg + np.stack([m, m, m], axis=-1)

    # And multiply by 255 to put them in the channels in the [0, 255] range
    rgbImg = rgbImg * 255

    # Convert to unsigned int8
    rgbImg = np.asarray(rgbImg, dtype=np.uint8)

    # And reshape back into the original shape
    rgbImg = rgbImg.reshape(originalShape)
    
    # Done!
    return rgbImg