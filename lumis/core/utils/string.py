import random
import re
import string


def number_to_words(n: int) -> str:
    """
    Converts a number to its string representation in English.

    Args:
        n (int): The number to be converted. Must be between 0 and 99 inclusive.

    Returns:
        str: The string representation of the number in English.

    Raises:
        ValueError: If the number is out of the range [0, 99].

    Examples:
        >>> number_to_words(10)
        'ten'
        >>> number_to_words(5)
        'five'
        >>> number_to_words(42)
        'forty two'
        >>> number_to_words(99)
        'ninety nine'
        >>> number_to_words(0)
        'zero'
    """
    if not (0 <= n < 100):
        raise ValueError("Number out of range, should be between 0 and 99")

    units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = [
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]
    tens = [
        "",
        "",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    ]

    if n == 0:
        return "zero"
    elif 1 <= n < 10:
        return units[n]
    elif 10 <= n < 20:
        return teens[n - 10]
    elif 20 <= n < 100:
        return tens[n // 10] + (" " + units[n % 10] if (n % 10 != 0) else "")
    else:
        return ""


def get_random_string(size=6, chars=string.ascii_uppercase + string.digits):
    """
    Generate a random string of the specified size.

    Parameters:
    size (int): The size of the random string to generate.

    Returns:
    str: A random string of the specified size.
    """

    result_str = "".join(random.choice(chars) for i in range(size))
    return result_str


def snake_to_title(snake_str: str) -> str:
    # Split the snake_case string by underscores
    words = snake_str.split("_")
    # Capitalize the first letter of each word and join them with a space
    return " ".join(word.capitalize() for word in words)
