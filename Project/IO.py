import os


LOGS_FILEPATH = os.getcwd() + "/logs/HeatMapGenerator.log"


def print_and_log(message):
    print(message)
    log_message(message)


def log_message(message):
    message += "\n"
    with open(LOGS_FILEPATH, 'a', errors="ignore") as output_file:
        output_file.writelines(message)


def create_log_file():
    """Creates directory and file for logs if it doesn't exist yet."""

    # Creates directory
    os.mkdir(LOGS_FILEPATH.rsplit('/', 1)[0] + '/')

    # Creating a file at specified location
    with open(LOGS_FILEPATH, 'w'):
        pass


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
