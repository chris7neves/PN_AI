import os
from datetime import datetime


class Logging:

    def __init__(self, file_name):
        self.path = os.path.dirname(os.path.realpath(__file__)) + "/logs/" + file_name + '_' + datetime.now().date()
        ensure_filepath_exists(self.path)

    def print_and_log(self, message):
        print(message)
        self.log_message(message)

    def log_message(self, message):
        message += "\n"
        with open(self.path, 'a', errors="ignore") as output_file:
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
