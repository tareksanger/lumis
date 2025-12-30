import hashlib


class ColorPrinter:
    @classmethod
    def hash_to_color(cls, value: str) -> int:
        """
        Generate a unique color code from a hash value.
        """
        hash_object = hashlib.md5(value.encode())
        hash_hex = hash_object.hexdigest()
        return int(hash_hex, 16) % 256

    @classmethod
    def generate_unique_color(cls, unique_id: str) -> str:
        """
        Generate a unique print color based on the unique ID.
        """
        color_code = cls.hash_to_color(unique_id) % 256  # There are 8 standard foreground colors
        return f"\033[38;5;{color_code}m"

    @classmethod
    def print(cls, text: str, unique_id: str) -> None:
        """
        Print the provided text in a unique color based on the unique ID.
        """
        color = cls.generate_unique_color(unique_id)
        reset_color = "\033[0m"  # Reset color to default
        print(f"{color}{text}{reset_color}")
