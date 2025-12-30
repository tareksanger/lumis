def format_number(num):
    """
    Formats a number with appropriate unit suffix (K, M, B, T) based on its magnitude.
    Only shows the decimal if the number is not whole.

    Args:
        num (float): The number to be formatted.

    Returns:
        str: The formatted number with unit suffix.

    Example:
        >>> format_number(1234567)
        '1.2M'
        >>> format_number(9876543210)
        '9.8B'
        >>> format_number(1000)
        '1K'
    """
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000.0:
            if num % 1 == 0:
                # Number is whole
                return f"{int(num)}{unit}"
            else:
                # Number is not whole
                return f"{num:.1f}{unit}"
        num /= 1000.0
    # For numbers larger than a trillion, check again if it's whole or not.
    if num % 1 == 0:
        return f"{int(num)}T"
    else:
        return f"{num:.1f}T"
