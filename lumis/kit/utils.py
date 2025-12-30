from __future__ import annotations

def dict_diff(d1: dict, d2: dict) -> dict:
    """
    Compute the difference between two nested dictionaries.
    Returns a dictionary containing the differences.
    """
    diff = {}
    keys = set(d1.keys()).union(d2.keys())
    for key in keys:
        v1 = d1.get(key, None)
        v2 = d2.get(key, None)
        if isinstance(v1, dict) and isinstance(v2, dict):
            sub_diff = dict_diff(v1, v2)
            if sub_diff:  # Only add if there's a difference
                diff[key] = sub_diff
        elif v1 != v2:
            # Return the new value
            diff[key] = v2
    return diff
