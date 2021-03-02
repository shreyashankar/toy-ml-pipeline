import random
import string
import time


def get_random_string(length: int) -> str:
    """
    This function returns a random string of letters of a specified length.

    Args:
        length (int): length of the random string to generate

    Returns:
        rand_str (str): Random string.
    """
    candidates = string.ascii_letters + string.digits
    rand_str = ''.join(random.choice(candidates)
                       for i in range(length))
    return rand_str


def get_timestamp_as_string() -> str:
    """
    This function returns a timestamp as a writable string.

    Returns:
        ts (str): String with the current time and date format as %Y%m%d-%H%M%S.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    return ts
