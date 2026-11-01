"""
Domain adaptation utilities for cross-device and cross-dataset generalization.
Includes histogram matching and other preprocessing techniques.
"""

import numpy as np
import cv2
from typing import Optional
import torch
from skimage import exposure


def histogram_matching(
    source: np.ndarray,
    reference: np.ndarray,
    multichannel: bool = True
) -> np.ndarray:
    """
    Match the histogram of a source image to a reference image.

    This technique helps normalize images from different devices/datasets
    to reduce domain shift.

    Args:
        source: Source image to transform (H, W, C) or (H, W)
        reference: Reference image with target distribution
        multichannel: If True, process each channel independently

    Returns:
        Histogram-matched image
    """
    if multichannel and source.ndim == 3:
        matched = np.zeros_like(source)
        for i in range(source.shape[2]):
            matched[:, :, i] = exposure.match_histograms(
                source[:, :, i],
                reference[:, :, i] if reference.ndim == 3 else reference
            )
        return matched
    else:
        return exposure.match_histograms(source, reference)


def clahe_enhancement(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Enhances local contrast in retinal images, making vessels and
    pathological features more visible.

    Args:
        image: Input image (H, W, C) or (H, W)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        Enhanced image
    """
    if image.ndim == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l = clahe.apply(l)

        # Merge and convert back to RGB
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)


def green_channel_extraction(image: np.ndarray) -> np.ndarray:
    """
    Extract green channel from RGB retinal image.

    The green channel typically has the best contrast for retinal vessels.

    Args:
        image: RGB image (H, W, 3)

    Returns:
        Green channel (H, W)
    """
    if image.ndim == 3 and image.shape[2] == 3:
        return image[:, :, 1]
    return image


def apply_domain_adaptation(
    image: np.ndarray,
    reference: Optional[np.ndarray] = None,
    use_clahe: bool = True,
    use_histogram_matching: bool = False
) -> np.ndarray:
    """
    Apply domain adaptation preprocessing pipeline.

    Args:
        image: Input image
        reference: Reference image for histogram matching (optional)
        use_clahe: Apply CLAHE enhancement
        use_histogram_matching: Apply histogram matching to reference

    Returns:
        Preprocessed image
    """
    processed = image.copy()

    # Apply CLAHE for enhancement
    if use_clahe:
        processed = clahe_enhancement(processed)

    # Apply histogram matching if reference provided
    if use_histogram_matching and reference is not None:
        processed = histogram_matching(processed, reference)

    return processed


class DomainAdaptationTransform:
    """
    Callable transform for domain adaptation preprocessing.
    Can be used in data loading pipelines.
    """

    def __init__(
        self,
        use_clahe: bool = True,
        use_histogram_matching: bool = False,
        reference_image: Optional[np.ndarray] = None
    ):
        self.use_clahe = use_clahe
        self.use_histogram_matching = use_histogram_matching
        self.reference_image = reference_image

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return apply_domain_adaptation(
            image,
            reference=self.reference_image,
            use_clahe=self.use_clahe,
            use_histogram_matching=self.use_histogram_matching
        )
