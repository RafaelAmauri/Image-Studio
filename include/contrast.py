import numpy as np


def contrast_boost(img: np.typing.NDArray, boost: int):
    img = img.copy().astype(np.float32)