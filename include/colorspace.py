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

    hsvImg = []

    # The specific formula for converting from RGB to HSV is in https://en.wikipedia.org/wiki/HSL_and_HSV#From_RGB
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            # For a given pixel, find out the largest and smallest value across the R, G and B channels
            Cmax = np.max(img[row][column])
            Cmin = np.min(img[row][column])
            
            # And find out from which channel the largest value comes from
            CmaxIdx = np.argmax(img[row][column])
            # The difference between the largest and the smallest value across the R, G and B channels.
            delta = Cmax - Cmin

            # If the largest value comes from the Red channel
            if CmaxIdx == 0:
                if delta != 0:
                    hValue = 60 * ((img[row][column][1] - img[row][column][2]) / delta % 6) 
                    sValue = delta / Cmax if Cmax != 0 else 0
                else:
                    hValue = 0
                    sValue = 0
                vValue = Cmax

            # If it comes from the Green channel
            elif CmaxIdx == 1:
                if delta != 0:
                    hValue = 60 * ((img[row][column][2] - img[row][column][0]) / delta + 2)
                    sValue = delta / Cmax if Cmax != 0 else 0
                else:
                    hValue = 0
                    sValue = 0
                vValue = Cmax

            # Or if it comes from the Blue channel
            elif CmaxIdx == 2:
                if delta != 0:
                    hValue = 60 * ((img[row][column][0] - img[row][column][1]) / delta + 4)
                    sValue = delta / Cmax if Cmax != 0 else 0
                else:
                    hValue = 0
                    sValue = 0
                vValue = Cmax
            
            
            hsvImg.append([hValue, sValue, vValue])

    # Right now our hsvImg array is a huge one-dimension array. We have to reshape it so it is in the [H, W, C] format.
    # Conviniently, img.shape is already in that format, so we just reuse that.
    hsvImg = np.reshape(hsvImg, img.shape)


    return hsvImg