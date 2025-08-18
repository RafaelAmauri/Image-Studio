import numpy as np


def contrast_boost(img: np.typing.NDArray, boost: float) -> np.typing.NDArray:
    """Boosts the contrast in RGB images

    Args:
        img (np.typing.NDArray): The image (must be numpy array)
        boost (int): The boost percentage. Must be between 0 and 100

    Returns:
        img: The image with boosted contrast
    """
    img = img.copy().astype(np.float32)

    # Divided by two because the boost is divided between the lowtones and hightones.
    # For example, if boost = 5%, 2.5% goes to the lowtones and 2.5% to the hightones. This way, the boost can be
    # from 0 to 100. If I didn't divide by two, if boost = 100, then the lowtones and hightones would overlap XD.
    boost = boost / 2
    
    for channel in range(img.shape[-1]):
        lowtones  = np.percentile(img[..., channel], boost)
        hightones = np.percentile(img[..., channel], 100-boost)

        lowtones_mask  = img[..., channel] <= lowtones
        hightones_mask = img[..., channel] >= hightones

        img[..., channel][lowtones_mask]  = 0
        img[..., channel][hightones_mask] = 255

        midtones_mask = ~(lowtones_mask | hightones_mask)

        img[..., channel][midtones_mask] = (img[..., channel][midtones_mask] - lowtones) / (hightones - lowtones) * 255
        
    
    img = img.clip(0, 255).astype(np.uint8)

    return img