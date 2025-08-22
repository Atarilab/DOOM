import ast
import re
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import yaml
import numpy as np

# Define the time format
time_format = "%Y-%m-%d %H:%M:%S,%f"

# Constants
MASS = 15.019  # Mass of the robot in kg


def get_cot(state):
    """
    Calculate the Cost of Transport (COT) from the robot state.

    :param state: The robot's state dictionary.
    :return: COT value.
    """
    # Extract the necessary values from the state
    joint_vel = np.array(state["joint_vel"])
    joint_tau_est = np.array(state["joint_tau_est"])
    lin_vel_b = np.array(state["lin_vel_b"])

    # Calculate power: sum of joint_vel * joint_tau_est
    power = np.sum(np.abs(joint_vel * joint_tau_est))

    # Calculate speed: norm of the linear velocity in 2D
    speed = np.linalg.norm(
        lin_vel_b[:2]
    )  # Taking the first two components for 2D speed

    # Calculate mechanical COT
    cot = power / (
        9.81 * speed * MASS + 1e-6
    )  # Add small value to avoid division by zero

    return cot.item() if cot < 50.0 else 0.0


# Regex to detect the *start* of a new log entry
log_start_pattern = re.compile(
    r"^\[(?P<level>\w+)\] \[(?P<time>[^\]]+)\] \[(?P<file>[^:]+):(?P<line>\d+)\]: (?P<message>.*)$"
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
        if msg.startswith("LOC-FROM-VID-PAPER-EXP-01"):
            try:
                key, val = msg.split(": ", 1)
                log[key.strip()] = ast.literal_eval(val)
            except Exception:
                raise Exception("Exception")

    return parsed_logs


def plot(data, metric, output_dir, target=None, x_values_for_vlines=None):
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker="o", linestyle="-", color="b")
    if target is not None:
        plt.plot(target, linestyle="-", color="r")
    if x_values_for_vlines is not None:
        max_value = max(data)
        for x in x_values_for_vlines:
            if 0 <= x < len(data):  # Ensure x is within bounds
                plt.vlines(x, 0, max_value, colors="g", lw=2)
    plt.xlabel("Log Message Index")
    plt.ylabel(metric)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{output_dir}/{metric}.pdf")
    plt.close()

    print(f"Saved plot {output_dir}/{metric}.pdf")


# Example usage
if __name__ == "__main__":
    base_folder = "src/logs/"
    experiment_string = "locFromVid"
    target_file_name = "rl-velocity-sim-go2_robot_controller.log"
    data_key = "LOC-FROM-VID-PAPER-EXP-01"

    pattern = re.compile(r"locFromVid_x(-?\d+\.?\d*)y(-?\d+\.?\d*)yaw(-?\d+\.?\d*)")

    # Use os.walk to recursively search for files
    for root, dirs, files in os.walk(base_folder):
        # Check if the current directory name starts with 'locFromVid' and the file exists in it
        if (
            os.path.basename(root).startswith(experiment_string)
            and target_file_name in files
        ):
            file_path = os.path.join(root, target_file_name)
            print(f"Processing file: {file_path}")

            logs = parse_logs(file_path)

            # filter only log messages relevant for the paper
            filtered_logs = [log for log in logs if data_key in log["message"]]
            logs = filtered_logs

            ###################### METRICS
            # Experiment target values are contained in folder name
            match = pattern.search(os.path.basename(root))
            if match:
                target_x = float(match.group(1))
                target_y = float(match.group(2))
                target_yaw = float(match.group(3))

                print(
                    f"Processing experiment with: x={target_x}, y={target_y}, yaw={target_yaw}"
                )
            else:
                print(
                    f"Processing experiment in {os.path.basename(root)} with unknown parameters."
                )

            # Find where the target command is originally supplied
            # NOTE target commands in x,y,yaw can be set sequentially, ie they dont have to be set at the same time
            # NOTE we assume, the correct target command gets set only once, and the first time the are set is taken
            start_index = 0
            for i, log in enumerate(logs):
                current_target_x = 0.0
                current_target_y = 0.0
                current_target_yaw = 0.0
                if log.get("LOC-FROM-VID-PAPER-EXP-01 Command Updated"):
                    command_updated = log["LOC-FROM-VID-PAPER-EXP-01 Command Updated"]
                    # Get new command
                    if "x_velocity" in command_updated:
                        current_target_x = command_updated["x_velocity"]
                    if "y_velocity" in command_updated:
                        current_target_x = command_updated["y_velocity"]
                    if "yaw_rate" in command_updated:
                        current_target_x = command_updated["yaw_rate"]

                    # Check if all target commands are set as expected
                    if (
                        current_target_x == target_x
                        and current_target_y == target_y
                        and current_target_yaw == target_yaw
                    ):
                        start_index = i + 1
                        break
            assert not start_index == 0, (
                "Correct target command most likely not found. Check log folder naming and target commands."
            )

            start_time = datetime.strptime(logs[start_index]["time"], time_format)

            # find end and offset indexes
            trajectory_offset_metrics = 2  # seconds
            trajectory_length_metrics = 5  # seconds
            offset_index = 0
            end_index = 0
            for i, log in enumerate(logs[start_index:]):
                time_difference = (
                    datetime.strptime(log["time"], time_format) - start_time
                )
                time_difference = time_difference.total_seconds()
                if (
                    time_difference > float(trajectory_offset_metrics)
                    and offset_index == 0
                ):
                    offset_index = (
                        i + start_index
                    )  # this is the index wrt the original logs list

                if time_difference > float(trajectory_length_metrics):
                    end_index = (
                        i + start_index
                    )  # this is the index wrt the original logs list
                    break

                # check that command does not get updated
                assert "LOC-FROM-VID-PAPER-EXP-01 Command Updated" not in log, (
                    "Target command has been changed during metric evaluation."
                )

            offset_time = datetime.strptime(logs[offset_index]["time"], time_format)
            end_time = datetime.strptime(logs[end_index]["time"], time_format)

            logs_metrics = logs[start_index + offset_index : end_index]

            print(
                f"Trajectory start index: {start_index}, offset index: {offset_index}, end index: {end_index}"
            )
            print(
                f"Trajectory start time: {start_time}, offset time: {offset_time}, end time: {end_time}"
            )
            print(
                f"Metric trajectory duration (offset -> end): {(end_time - offset_time)}"
            )
            print(f"Metric trajectory num logs: {end_index - offset_index}")
            print(
                f"Average time between logs: {(end_time - offset_time) / (end_index - offset_index)}"
            )
            print("\n")

            ###################### PLOTS
            cots = []
            vels_x = []
            vels_y = []
            yaws = []

            target_x = []
            target_y = []
            target_yaw = []

            current_target_x = 0.0
            current_target_y = 0.0
            current_target_yaw = 0.0

            for log in logs:
                if log.get("LOC-FROM-VID-PAPER-EXP-01 Command Updated"):
                    command_updated = log["LOC-FROM-VID-PAPER-EXP-01 Command Updated"]
                    if "x_velocity" in command_updated:
                        current_target_x = command_updated["x_velocity"]
                    if "y_velocity" in command_updated:
                        current_target_y = command_updated["y_velocity"]
                    if "yaw_rate" in command_updated:
                        current_target_yaw = command_updated["yaw_rate"]

                if log.get("LOC-FROM-VID-PAPER-EXP-01"):
                    state = log["LOC-FROM-VID-PAPER-EXP-01"]["state"]
                    cot = get_cot(state)
                    cots.append(cot)
                    vels_x.append(state["lin_vel_b"][0])
                    vels_y.append(state["lin_vel_b"][1])
                    yaws.append(state["gyroscope"][2])

                    target_x.append(current_target_x)
                    target_y.append(current_target_y)
                    target_yaw.append(current_target_yaw)

            x_values_for_vlines = [start_index, offset_index, end_index]
            output_dir = f"paper_locFromVid/outputs/{os.path.basename(root)}"
            os.makedirs(output_dir, exist_ok=True)

            plot(cots, "cot", output_dir, None, x_values_for_vlines)
            plot(vels_x, "x vel", output_dir, target_x, x_values_for_vlines)
            plot(vels_y, "y vel", output_dir, target_y, x_values_for_vlines)
            plot(yaws, "yaw", output_dir, target_yaw, x_values_for_vlines)

            ###################### VALUES AND METRICS
            # NOTE this calculation is not entirely correct, as the indexes for the command changes are neglected

            cots = cots[offset_index:end_index]
            vels_x = vels_x[offset_index:end_index]
            vels_y = vels_y[offset_index:end_index]
            yaws = yaws[offset_index:end_index]

            values = {
                "cots": cots,
                "vels_x": vels_x,
                "vels_y": vels_y,
                "yaws": yaws,
            }
            
            metrics = {
                "mean_cots": np.mean(np.array(cots)).item(),
                "mean_vels_x": np.mean(np.array(vels_x)).item(),
                "mean_vels_y": np.mean(np.array(vels_y)).item(),
                "mean_yaws": np.mean(np.array(yaws)).item(),
                "std_cots": np.std(np.array(cots)).item(),
                "std_vels_x": np.std(np.array(vels_x)).item(),
                "std_vels_y": np.std(np.array(vels_y)).item(),
                "std_yaws": np.std(np.array(yaws)).item(),
            }
            
            # Save data to a YAML file
            with open(os.path.join(output_dir, "metrics.yaml"), "w") as file:
                yaml.dump(metrics, file)
                            # Save data to a YAML file
            with open(os.path.join(output_dir, "values.yaml"), "w") as file:
                yaml.dump(values, file)

            print(f"\nData saved to {os.path.join(output_dir, 'metrics.yaml')}")
