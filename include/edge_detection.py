import numpy as np
import warnings

import include.colormodel as colormodel
import include.convolve2d as convolve2d
import include.kernels as kernels


def sobel(img: np.typing.ArrayLike, edgeColor: int) -> np.typing.ArrayLike:
    """
    Implements Sobel edge detection https://en.wikipedia.org/wiki/Sobel_operator.

    Works only with grayscale images. If the input image is RGB, the function automatically converts it
    to grayscale.

    Args:
        img (np.typing.ArrayLike): The image.

        edgeColor (int) : The HSV color to use to color the edges. -2 = White edges, -1 = Uses the edge direction
        in an HSV color wheel (https://i.sstatic.net/UyDZ8.jpg) to automatically get the edge color. 
        Any other number will use the same HSV color wheel to choose a color and then color all 
        edges with that color.

    Returns:
        np.typing.ArrayLike: The image with the detected edges in RGB format.
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
    gradientHorizontalEdges = convolve2d.convolve2d(img, kernels.sobelHorizontal3x3, "edge")
    gradientVerticalEdges   = convolve2d.convolve2d(img, kernels.sobelVertical3x3,   "edge")

    # Combine horizontal and vertical edges
    gradient = np.sqrt((gradientHorizontalEdges**2) + (gradientVerticalEdges**2))

    # Treat the horizontal and vertical gradients as a right triangle and use the arctangent
    # of both gradients to get the direction for the edges.
    gradientDirection = np.atan2(gradientHorizontalEdges, gradientVerticalEdges)
    # Convert from radians to degrees
    gradientDirection = np.rad2deg(gradientDirection) % 360

    # Normalize the pixel values to the range [0, 1]
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
    
    if edgeColor == -2:
        img = np.stack([gradientDirection, np.full_like(gradientDirection, 0.0), gradient], axis=2)
    elif edgeColor == -1:
        img = np.stack([gradientDirection, np.full_like(gradientDirection, 0.8), gradient], axis=2)
    else:
        img = np.stack([np.full_like(gradientDirection, edgeColor), np.full_like(gradientDirection, 0.8), gradient], axis=2)
    
    img = colormodel.hsv2rgb(img)

    return img


def prewitt(img: np.typing.ArrayLike, edgeColor: int) -> np.typing.ArrayLike:
    """
    Implements Prewitt edge detection https://en.wikipedia.org/wiki/Prewitt_operator.

    Works only with grayscale images. If the input image is RGB, the function automatically converts it
    to grayscale.

    Args:
        img (np.typing.ArrayLike): The image.

        edgeColor (int) : The HSV color to use to color the edges. -2 = White edges, -1 = Uses the edge direction
        in an HSV color wheel (https://i.sstatic.net/UyDZ8.jpg) to automatically get the edge color. 
        Any other number will use the same HSV color wheel to choose a color and then color all 
        edges with that color.

    Returns:
        np.typing.ArrayLike: The image with the detected edges in RGB format.
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
    gradientHorizontalEdges = convolve2d.convolve2d(img, kernels.prewittHorizontal3x3, "edge")
    gradientVerticalEdges   = convolve2d.convolve2d(img, kernels.prewittVertical3x3,   "edge")

    # Combine horizontal and vertical edges
    gradient = np.sqrt((gradientHorizontalEdges**2) + (gradientVerticalEdges**2))

    # Treat the horizontal and vertical gradients as a right triangle and use the arctangent
    # of both gradients to get the direction for the edges.
    gradientDirection = np.atan2(gradientHorizontalEdges, gradientVerticalEdges)
    # Convert from radians to degrees
    gradientDirection = np.rad2deg(gradientDirection) % 360

    # Normalize the pixel values to the range [0, 1]
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
    
    # Depending on what value edgeColor is, we use different
    if edgeColor == -2:
        img = np.stack([gradientDirection                         , np.full_like(gradientDirection, 0.0), gradient], axis=2)
    elif edgeColor == -1:
        img = np.stack([gradientDirection                         , np.full_like(gradientDirection, 0.8), gradient], axis=2)
    else:
        img = np.stack([np.full_like(gradientDirection, edgeColor), np.full_like(gradientDirection, 0.8), gradient], axis=2)
    
    img = colormodel.hsv2rgb(img)

    return img