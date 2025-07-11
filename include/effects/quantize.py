import numpy as np

def nearestColor(pixelColor: np.typing.NDArray, availableColors: np.typing.NDArray) -> np.typing.NDArray:
    """
    Given a list of available colors, picks the one closest to pixelColor

    Args:
        pixelColor (int): The original color
        availableColors (np.typing.NDArray): The list of available colors

    Returns:
        int: The color in availableColors closest to pixelColor
    """
    candidate1Idx = np.searchsorted(availableColors, pixelColor, "right") - 1
    candidate2Idx = np.clip(candidate1Idx + 1, 0, len(availableColors)-1)
    
    candidate1 = availableColors[candidate1Idx]
    candidate2 = availableColors[candidate2Idx]
    pixelColor = pixelColor
    
    # The new pixel color is whichever of the two candidate colors are the closest to it
    quantizedColor = np.where(
                        (pixelColor - candidate1) < (candidate2 - pixelColor),
                        candidate1,
                        candidate2
    )
    
    return quantizedColor


def quantize(img: np.typing.NDArray, availableColors: np.typing.NDArray) -> np.typing.NDArray:
    """Quantizes the image into an arbitrary number of colors.

    Args:
        img (np.typing.NDArray)             : The image array. Must be in the format (H, W, C)
        availableColors (np.typing.NDArray) : A list containing the colors available. Should start at 0 and 
                                                the last element should be 255.
    Returns:
        np.typing.NDArray (np.uint8): The quantized image
    """
    originalImgShape = img.shape
    img = img.copy()

    # Reshape img into (H * W, 3)
    img = img.reshape(-1, img.shape[-1])
    
    # For each channel, pass it as an argument to nearestColor. Since nearestColor is fully vectorized, it
    # applies the function to the pixels in each channel separately
    img  = np.stack([nearestColor(img[:, channel], availableColors) for channel in range(img.shape[-1])], axis=1)
    
    # Reshape the image back into the original shape
    img = img.reshape(originalImgShape)
    

    img = img.astype(np.uint8)
    return img