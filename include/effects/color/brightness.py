import numpy as np


def brightness_boost(img: np.typing.NDArray, boost: int) -> np.typing.NDArray:
    """Boosts the brightness in RGB images

    Args:
        img (np.typing.NDArray): The image (must be numpy array)
        boost (int): The boost percentage. Must be between -255 and 255

    Returns:
        img: The image with boosted brightness
    """
    img = img.copy().astype(np.int16)

    img = (img + boost).clip(0, 255).astype(np.uint8)

    return img