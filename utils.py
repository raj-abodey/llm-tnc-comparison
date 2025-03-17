# Utility functions
from typing import Tuple


def read_tnc_files() -> Tuple[str, str]:
    """Utility function to read the t&c files

    Returns:
        Tuple[str, str]: t&c for 2015 and 2023
    """
    file_path1 = "./data/Jan2015.txt"
    file_path2 = "./data/Mar2023.txt"
    with open(file_path1, "r") as file:
        jan_2015_tnc = file.read()
    with open(file_path2, "r") as file:
        mar_2023_tnc = file.read()

    return jan_2015_tnc, mar_2023_tnc
