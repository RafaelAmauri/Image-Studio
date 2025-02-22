import numpy as np
import warnings

import include.colormodel as colormodel
import include.convolve2d as convolve2d
import include.kernels as kernels


def sobel(img: np.typing.ArrayLike) -> np.typing.ArrayLike:
    """
    Implements Sobel edge detection https://en.wikipedia.org/wiki/Sobel_operator.

    Works only with grayscale images. If the input image is RGB, the function automatically converts it
    to grayscale.

    Args:
        img (np.typing.ArrayLike): The image.

    Returns:
        np.typing.ArrayLike: The image with the detected edges.
    """
    # If it's not a grayscale image
    if img.shape[2] != 1:
        warnings.warn("Cannot do edge detection on an RGB image! Automatically converting to grayscale...\n"\
              "Expect weird results, especially if the image is quantized, because now the edge detection "\
              "will mark the color banding artifacts as edges! Consider using the -g option.")
        img = colormodel.rgb2grayscale(img)

    # Remove the fake 'channel' dimension
    img = img.squeeze(axis=2)

    # Directly cast to float32
    img = img.astype(np.float32)

    # Calculate edges
    horizontalEdges = convolve2d.convolve2d(img, kernels.sobelHorizontal3x3)
    verticalEdges   = convolve2d.convolve2d(img, kernels.sobelVertical3x3)

    # Combine horizontal and vertical edges
    edges = np.sqrt((horizontalEdges**2) + (verticalEdges**2))

    # Normalize the pixel values to the range [0, 255]
    edges = (edges - edges.min()) / (edges.max() - edges.min()) * 255

    # Add back the fake 'channel' dimension and convert to uint8
    edges = np.expand_dims(edges, axis=2).astype(np.uint8)

    return edges


def prewitt(img: np.typing.ArrayLike) -> np.typing.ArrayLike:
    """
    Implements Prewitt edge detection (https://en.wikipedia.org/wiki/Prewitt_operator).

    Works only with grayscale images. If the input image is RGB, the function automatically converts it
    to grayscale.

    Args:
        img (np.typing.ArrayLike): The image.

    Returns:
        np.typing.ArrayLike: The image with the detected edges.
    """

    # If it's not a grayscale image
    if img.shape[2] != 1:
        warnings.warn("Cannot do edge detection on an RGB image! Automatically converting to grayscale...\n"\
              "Expect weird results, especially if the image is quantized, because now the edge detection "\
              "will mark the color banding artifacts as edges! Consider using the -g option.")
        img = colormodel.rgb2grayscale(img)

    # Remove the fake 'channel' dimension
    img = img.squeeze(axis=2)

    # Directly cast to float32
    img = img.astype(np.float32)

    # Calculate edges
    horizontalEdges = convolve2d.convolve2d(img, kernels.prewittHorizontal3x3)
    verticalEdges   = convolve2d.convolve2d(img, kernels.prewittVertical3x3)

    # Combine horizontal and vertical edges
    edges = np.sqrt((horizontalEdges**2) + (verticalEdges**2))

    # Normalize the pixel values to the range [0, 255]
    edges = (edges - edges.min()) / (edges.max() - edges.min()) * 255

    # Add back the fake 'channel' dimension and convert to uint8
    edges = np.expand_dims(edges, axis=2).astype(np.uint8)

    return edges