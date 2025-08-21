import argparse
import ast
import re

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
    logs = parse_logs(args.file_path)
    for log in logs[:]:
        print(log, "\n")
