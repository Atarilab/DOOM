import logging
import os


def get_logger(
    name,
    log_file="system.log",
    debug=False,
    file_level=logging.DEBUG,
    console_level=logging.INFO,
):
    """
    Create and configure a logger with file and console handlers.

    Args:
        name: Name of the logger
        log_file: Path to the log file
        debug: Whether to enable debug mode
        file_level: Logging level for file handler
        console_level: Logging level for console handler

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to lowest level to capture all logs

    if logger.hasHandlers():
        logger.handlers.clear()

    log_format = (
        "[%(levelname)s] [%(asctime)s] "
        "[%(filename)s:%(lineno)d]: %(message)s"
    )
    # log_format = '[%(asctime)s] : %(message)s'

    # File Logs
    try:
        final_log_file = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", log_file))
        log_dir = os.path.dirname(final_log_file)

        os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(final_log_file, mode="w")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to create log file handler: {e}")

    # Console Logs
    console_level = logging.INFO if not debug else logging.DEBUG
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    return logger
