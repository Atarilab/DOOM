from utils.logger import logging
import time

# Add ROS2 Vicon subscriber
def vicon_handler(msg, logger: logging.Logger):
    # print("Raw message:", msg)
    # print("Message type:", type(msg))
    
    # try:
    #     # Attempt to print each field
    #     for field in msg.get_fields_and_field_types().keys():
    #         print(f"{field}: {getattr(msg, field)}")
    # except Exception as e:
    #     print(f"Error accessing message fields: {e}")
    
    logger.debug(f"Received Vicon data at {time.time()}")
    
# Add Low State Subscriver
def low_state_handler(msg, logger: logging.Logger):
    logger.debug(f"Received low state at {time.time()}")
    
    # try:
    #     # Log detailed message inspection
    #     print("FR_0 motor state: ", msg.motor_state[go2.LegID["FR_0"]].q)
    # except Exception as e:
    #     logger.error(f"Error processing low state: {e}")