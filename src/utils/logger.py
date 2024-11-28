import os
import logging

def get_logger(name, log_file="system.log", debug=False):
    level = logging.INFO if not debug else logging.DEBUG
    
    # Ensure the log directory exists
    log_dir = os.path.dirname(log_file)
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create log directory: {e}")
        # Fallback to current directory
        log_dir = '.'
        log_file = os.path.join(log_dir, os.path.basename(log_file))

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    except IOError as e:
        print(f"Failed to create log file handler: {e}")
        # Ensure console logging still works

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    return logger