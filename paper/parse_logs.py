import argparse
import ast
import re
import numpy as np
import matplotlib.pyplot as plt

# Constants
MASS = 15  # Mass of the robot in kg

def calculate_cot(state):
    """
    Calculate the Cost of Transport (COT) from the robot state.

    :param state: The robot's state dictionary.
    :return: COT value.
    """
    # Extract the necessary values from the state
    joint_vel = np.array(state['joint_vel'])
    joint_tau_est = np.array(state['joint_tau_est'])
    lin_vel_b = np.array(state['lin_vel_b'])
    
    # Calculate power: sum of joint_vel * joint_tau_est
    power = np.sum(np.abs(joint_vel * joint_tau_est))
    
    # Calculate speed: norm of the linear velocity in 2D
    speed = np.linalg.norm(lin_vel_b[:2])  # Taking the first two components for 2D speed
    
    # Calculate mechanical COT
    cot = power / (9.81 * speed * MASS + 1e-6)  # Add small value to avoid division by zero
    
    
    return cot.item() if cot < 50.0 else 0.0

def calculate_cots_for_logs(logs):
    """
    Given a list of parsed logs, calculate the Cost of Transport (COT) for each log.

    :param logs: List of parsed logs, where each log is a dictionary with 'state' data.
    :return: List of Cost of Transport values for each log.
    """
    cot_values = []
    
    for log in logs:
        
        if log.get('OLIVER-PAPER-EXP-01') and log['OLIVER-PAPER-EXP-01'].get('state'):
        
            cot = calculate_cot(log['OLIVER-PAPER-EXP-01']['state'])
            cot_values.append(cot)
    
    return cot_values


def plot_cot(cot_values):
    """ Plot Cost of Transport over time. """
    plt.figure(figsize=(10, 5))
    plt.plot(cot_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Log Message Index')
    plt.ylabel('Cost of Transport')
    plt.title('Cost of Transport Over Time')
    plt.tight_layout()
    plt.savefig("test.pdf")


# Regex to detect the *start* of a new log entry
log_start_pattern = re.compile(
    r'^\[(?P<level>\w+)\] \[(?P<time>[^\]]+)\] \[(?P<file>[^:]+):(?P<line>\d+)\]: (?P<message>.*)$'
)

def parse_logs(filepath):
    parsed_logs = []
    current_log = None

    with open(filepath) as f:
        for line in f:
            line = line.rstrip("\n")

            # Check if this is a new log start
            m = log_start_pattern.match(line)
            if m:
                # If we were collecting a previous multi-line log, save it
                if current_log:
                    parsed_logs.append(current_log)

                current_log = m.groupdict()
                current_log["raw_message"] = [current_log["message"]]
            elif current_log:
                # Continuation of the previous log
                current_log["raw_message"].append(line)

        # Append the last one
        if current_log:
            parsed_logs.append(current_log)

    # Postprocess: merge multiline messages & try to parse dict-like content
    for log in parsed_logs:
        msg = "\n".join(log.pop("raw_message")).strip()
        log["message"] = msg

        # Try parsing dict-like message contents
        if msg.startswith("OLIVER-PAPER-EXP-01"):
            try:
                key, val = msg.split(": ", 1)
                log[key.strip()] = ast.literal_eval(val)
            except Exception:
                raise Exception("Exception")

    return parsed_logs


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse a log file and process its contents.')
    parser.add_argument('--file_path', type=str, help='The path to the log file to process')
    args = parser.parse_args()
    logs = parse_logs("src/logs/oliver/rl-velocity-sim-go2_robot_controller.log")#args.file_path)
    # Calculate COT values for the parsed logs
    cot_values = calculate_cots_for_logs(logs)

    # Plot the COT over time
    plot_cot(cot_values)
