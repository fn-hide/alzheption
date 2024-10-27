import cv2 as cv
import numpy as np



def calculate_sharpness(image, normalize=False, max_value=1000) -> float:
    """Sharpness can be assessed using image gradients or edge detection. One common method is to use the Laplacian operator.

    Args:
        image (_type_): _description_
        normalize (bool, optional): _description_. Defaults to False.
        max_value (int, optional): _description_. Defaults to 1000.

    Returns:
        float: _description_
    """
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    laplacian_var = cv.Laplacian(image, cv.CV_64F).var()
    
    if normalize:
        return min(laplacian_var / max_value, 1.0)
    return laplacian_var


def calculate_brightness(image, normalize=False) -> float:
    """Brightness is the average pixel intensity in grayscale.

    Args:
        image (_type_): _description_
        normalize (bool, optional): _description_. Defaults to False.

    Returns:
        float: _description_
    """
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    brightness = np.mean(image)
    
    if normalize:
        return brightness / 255.0
    return brightness


def calculate_contrast(image, normalize=False, max_value=100) -> float:
    """Contrast can be assessed by the standard deviation of pixel intensities.

    Args:
        image (_type_): _description_
        normalize (bool, optional): _description_. Defaults to False.
        max_value (int, optional): _description_. Defaults to 100.

    Returns:
        float: _description_
    """
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    contrast = image.std()
    
    if normalize:
        return min(contrast / max_value, 1.0)
    return contrast


def calculate_unique_intensity_levels(image, normalize=False, max_levels=256):
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    unique_levels = len(np.unique(image))
    
    if normalize:
        return unique_levels / max_levels
    return unique_levels


def calculate_shadow(image) -> float:
    """Shadow detection can be complex, but a simple approach is to look for very dark regions in the image.

    Args:
        image (_type_): _description_

    Returns:
        float: _description_
    """
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    dark_area_percentage = np.sum(image < 50) / image.size
    
    return 1 - min(dark_area_percentage, 1.0)


def calculate_specularity(image) -> float:
    """Specularity can be estimated by detecting bright spots.

    Args:
        image (_type_): _description_

    Returns:
        float: _description_
    """
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    bright_spots_percentage = np.sum(image > 200) / image.size
    
    return 1 - min(bright_spots_percentage, 1.0)


def calculate_background_uniformity(image) -> float:
    """Assess uniformity by comparing the variance of pixel intensities in the background.

    Args:
        image (_type_): _description_

    Returns:
        float: _description_
    """
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    background_region = image[:int(image.shape[0] / 4), :]  # Example region
    return 1 - min(np.std(background_region) / 255.0, 1.0)

