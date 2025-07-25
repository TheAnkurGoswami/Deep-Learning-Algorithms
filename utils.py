import numpy as np


def check_closeness(
    a: np.ndarray,
    b: np.ndarray,
    additional_checks: bool = True,
    tolerance: float = 1e-06,
) -> bool:
    """
    Check if two numpy arrays are close to each other within a certain
    tolerance.

    Parameters:
    a (np.ndarray): First array to compare.
    b (np.ndarray): Second array to compare.
    additional_checks (bool): If True, perform additional checks for closeness.
        Default is True.
    tolerance (float): Tolerance value for element-wise comparison.
        Default is 1e-06.

    Returns:
    bool: True if arrays are close to each other, False otherwise.
    """
    # Check if arrays are element-wise equal within a tolerance
    main_check = np.allclose(a, b)
    clip_const = 1e-06
    # # Reverse clip values of 'a' if they are of the order of e-07 or lower
    a = np.where(np.abs(a) > clip_const, a, clip_const)
    b = np.where(np.abs(b) > clip_const, b, clip_const)
    # Check if the absolute difference between arrays is within the tolerance
    other_check = np.abs(a - b) <= tolerance
    with np.errstate(divide="ignore", invalid="ignore"):
        # Calculate the minimum of the two arrays element-wise
        # min_arr = np.minimum(a, b)
        max_arr = np.maximum(a, b)
        clipped_diff = np.where(np.abs(a - b) > clip_const, np.abs(a - b), 0)
        # Calculate the percentage difference where max_arr is not zero
        percent_diff = np.average(
            np.where(max_arr != 0, clipped_diff / max_arr * 100, 0)
        )
        # Check if the average percentage difference is within 0.001%
        precent_check = percent_diff <= 0.001
    if additional_checks:
        # Return True if any of the checks pass
        return bool(main_check or np.all(other_check) or precent_check)

    # Return the result of the main check
    return main_check
