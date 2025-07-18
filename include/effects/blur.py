import numpy as np

import include.utils.convolve2d as convolve2d
import include.utils.kernels as kernels


blurKernels = {
                "boxblur3x3":  kernels.boxBlur3x3,
                "boxblur5x5":  kernels.boxBlur5x5,
                "gaussian3x3": kernels.gaussianBlur3x3,
                "gaussian5x5": kernels.gaussianBlur5x5
            }


def blur(img: np.typing.NDArray, kernelName: str) -> np.typing.NDArray:
    kernel = blurKernels[kernelName]

    img    = np.stack([convolve2d.convolve2d(img[..., channel], kernel) for channel in range(img.shape[-1])], axis=2)
    img    = img.astype(np.uint8)

    return img