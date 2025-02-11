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
    """Converts an image from the RGB colorspace into the HSV colorspace (https://en.wikipedia.org/wiki/HSL_and_HSV)!

    Args:
        img (np.typing.ArrayLike): The RGB image.

    Returns:
        np.typing.ArrayLike: The HSV image.
    """
    # Normalize the image
    img = img / 255

    originalImgShape = img.shape

    # This is a bit weird, but doing this enables me to convert an array of RGB values of any shape
    # into HSV value, and not just convert rgb images into hsv images. A neat application of this
    # is being able to convert an RGB color into an HSV color.
    img = np.reshape(img, (-1, 3))
    
    hsvImg = []

    # The specific formula for converting from RGB to HSV is in https://en.wikipedia.org/wiki/HSL_and_HSV#From_RGB
    for rgbValue in img:
        # For a given pixel, find out the largest and smallest value across the R, G and B channels
        Cmax = np.max(rgbValue)
        Cmin = np.min(rgbValue)
        
        # And find out from which channel the largest value comes from
        CmaxIdx = np.argmax(rgbValue)
        # The difference between the largest and the smallest value across the R, G and B channels.
        delta = Cmax - Cmin

        # If the largest value comes from the Red channel
        if CmaxIdx == 0:
            if delta != 0:
                hValue = 60 * ((rgbValue[1] - rgbValue[2]) / delta % 6) 
                sValue = delta / Cmax if Cmax != 0 else 0
            else:
                hValue = 0
                sValue = 0
            vValue = Cmax

        # If it comes from the Green channel
        elif CmaxIdx == 1:
            if delta != 0:
                hValue = 60 * ((rgbValue[2] - rgbValue[0]) / delta + 2)
                sValue = delta / Cmax if Cmax != 0 else 0
            else:
                hValue = 0
                sValue = 0
            vValue = Cmax

        # Or if it comes from the Blue channel
        elif CmaxIdx == 2:
            if delta != 0:
                hValue = 60 * ((rgbValue[0] - rgbValue[1]) / delta + 4)
                sValue = delta / Cmax if Cmax != 0 else 0
            else:
                hValue = 0
                sValue = 0
            vValue = Cmax
        
        
        hsvImg.append([hValue, sValue, vValue])
    
    hsvImg = np.asarray(hsvImg, dtype=np.float32)
    
    # Right now our hsvImg array is a huge one-dimension array. We have to reshape it so it is in the [H, W, C] format.
    # Conviniently, originalImgShape is already in that format, so we just reuse that.
    hsvImg = np.reshape(hsvImg, originalImgShape)


    return hsvImg