import os
from datetime import datetime


def print_and_log(logs_filepath, message):
    print(message)
    log_message(logs_filepath, message)


def log_message(logs_filepath, message):
    message += "\n"
    with open(logs_filepath, 'a', errors="ignore") as output_file:
        output_file.writelines(f"[{datetime.now()}] {message}")


def ensure_filepath_exists(file_path):
    """Creates directory and file for logs if it doesn't exist yet."""

    # Creates directory.
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Creates an empty file at a specified location.
    if not file_path.endswith("/") and not os.path.isfile(file_path):
        with open(file_path, 'w'):
            pass
