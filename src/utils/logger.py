import os
import logging

def get_logger(name, log_file="system.log", debug=False, file_level=logging.DEBUG, console_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to lowest level to capture all logs

    if logger.hasHandlers():
        logger.handlers.clear()

    log_format = '[%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s'

    # File Logs
    try:
        final_log_file = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '../', log_file)
        )
        log_dir = os.path.dirname(final_log_file)
        
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(final_log_file)
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