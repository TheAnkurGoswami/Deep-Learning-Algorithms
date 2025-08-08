# Utils

The `utils.py` file contains utility functions that are used throughout the codebase.

## `check_closeness`

This function checks if two numpy arrays, `a` and `b`, are close to each other.

`check_closeness(a, b, additional_checks=True, tolerance=1e-06)`

### Parameters

-   `a` (np.ndarray): The first array.
-   `b` (np.ndarray): The second array.
-   `additional_checks` (bool): If `True`, performs additional checks for closeness. Defaults to `True`.
-   `tolerance` (float): The tolerance for element-wise comparison. Defaults to `1e-06`.

### Returns

-   `bool`: `True` if the arrays are close to each other, `False` otherwise.

### Logic

1.  It first uses `np.allclose(a, b)` for a primary check.
2.  If `additional_checks` is `True`, it performs two more checks:
    -   Checks if the absolute difference between the arrays is within the given `tolerance`.
    -   Calculates the average percentage difference and checks if it's within `0.001%`.
3.  If any of these checks pass, it returns `True`.
4.  If `additional_checks` is `False`, it only returns the result of `np.allclose`.
