import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth_noise(noise: np.ndarray, kernel_std: float) -> np.ndarray:
    """Apply a Gaussian filter to smooth noise.

    Maintains mean and variance of input.

    Args:
        noise: (T, dU), input noise array
        kernel_std: Size of the Gaussian Kernel

    Returns:
        smoothed: (T, dU), smoothed noise array
    """
    _, dU = noise.shape

    # Apply Gaussian filter
    smoothed = np.empty_like(noise)
    for i in range(dU):
        smoothed[:, i] = gaussian_filter1d(noise[:, i], kernel_std, mode="nearest")

    # Renorm
    smoothed *= np.std(noise, axis=0) / np.std(smoothed, axis=0)
    smoothed += np.mean(noise, axis=0) - np.mean(smoothed, axis=0)
    return smoothed
