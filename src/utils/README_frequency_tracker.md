# FrequencyTracker Utility

A reusable utility for tracking and logging the frequency of recurring events, function calls, or any other periodic operations in your codebase.

## Features

- **Simple API**: Easy to integrate into existing code
- **Flexible logging**: Support for custom loggers and log levels
- **Statistics tracking**: Min/max/average frequency monitoring
- **Multi-tracker support**: Track multiple events simultaneously
- **Thread-safe**: Safe for use in multi-threaded environments

## Quick Start

### Basic Usage

```python
from utils.frequency_tracker import FrequencyTracker

# Create a tracker
tracker = FrequencyTracker("my_function", log_interval=5.0)

# In your function or loop
def my_function():
    # Your code here
    tracker.tick()  # Call this on each execution
```

### With Custom Logger

```python
import logging
from utils.frequency_tracker import FrequencyTracker

logger = logging.getLogger(__name__)
tracker = FrequencyTracker("data_processing", log_interval=2.0, logger=logger, log_level="info")

while processing:
    process_data()
    tracker.tick()
```

### Multiple Trackers

```python
from utils.frequency_tracker import MultiFrequencyTracker

# Create multi-tracker
multi_tracker = MultiFrequencyTracker(default_log_interval=5.0)

# Add different trackers
multi_tracker.add_tracker("sensor_read", log_interval=1.0)
multi_tracker.add_tracker("control_loop", log_interval=2.0)

# Use in your code
while running:
    read_sensors()
    multi_tracker.tick("sensor_read")
    
    control_loop()
    multi_tracker.tick("control_loop")
```

## API Reference

### FrequencyTracker

#### Constructor
```python
FrequencyTracker(name, log_interval=5.0, logger=None, log_level="info")
```

- `name`: Identifier for this tracker (used in log messages)
- `log_interval`: Time interval in seconds between frequency logs
- `logger`: Logger object to use (if None, uses print)
- `log_level`: Log level ("info", "debug", "warning", "error")

#### Methods

- `tick(count=1)`: Record an event occurrence
- `get_current_frequency()`: Get current frequency without logging
- `get_statistics()`: Get comprehensive statistics
- `reset_statistics()`: Reset all counters and statistics
- `set_log_interval(interval)`: Update logging interval
- `set_logger(logger, log_level)`: Update logger and log level

### MultiFrequencyTracker

#### Constructor
```python
MultiFrequencyTracker(default_log_interval=5.0, logger=None)
```

#### Methods

- `add_tracker(name, log_interval=None, logger=None, log_level="info")`: Add a new tracker
- `tick(name, count=1)`: Record an event for a specific tracker
- `get_tracker(name)`: Get a specific tracker
- `get_all_statistics()`: Get statistics for all trackers
- `reset_all_statistics()`: Reset all trackers
- `remove_tracker(name)`: Remove a tracker

## Example Output

```
[INFO] compute_lowlevelcmd frequency: 50.25 Hz (calls: 251, time: 5.00s)
[INFO] sensor_read frequency: 100.00 Hz (calls: 100, time: 1.00s)
[INFO] control_loop frequency: 25.50 Hz (calls: 51, time: 2.00s)
```

## Statistics

The tracker provides comprehensive statistics:

```python
stats = tracker.get_statistics()
print(f"Average frequency: {stats['average_frequency']:.2f} Hz")
print(f"Min frequency: {stats['min_frequency']:.2f} Hz")
print(f"Max frequency: {stats['max_frequency']:.2f} Hz")
print(f"Total calls: {stats['total_calls']}")
print(f"Total time: {stats['total_time']:.2f} seconds")
```

## Use Cases

- **Performance monitoring**: Track function call frequencies
- **Control loops**: Monitor control loop execution rates
- **Data processing**: Track data processing throughput
- **Sensor reading**: Monitor sensor sampling rates
- **Network operations**: Track network request frequencies
- **File I/O**: Monitor file operation frequencies

## Integration Examples

### In a Controller Class

```python
class MyController:
    def __init__(self):
        self.freq_tracker = FrequencyTracker("control_loop", log_interval=5.0, logger=self.logger)
    
    def control_loop(self):
        self.freq_tracker.tick()
        # Your control logic here
```

### In a Data Processing Pipeline

```python
class DataProcessor:
    def __init__(self):
        self.multi_tracker = MultiFrequencyTracker()
        self.multi_tracker.add_tracker("data_read", 1.0)
        self.multi_tracker.add_tracker("data_process", 2.0)
        self.multi_tracker.add_tracker("data_write", 1.5)
    
    def process_pipeline(self):
        self.multi_tracker.tick("data_read")
        # Read data
        
        self.multi_tracker.tick("data_process")
        # Process data
        
        self.multi_tracker.tick("data_write")
        # Write data
```

## Running Examples

To see the FrequencyTracker in action, run the example script:

```bash
cd src/utils
python frequency_tracker_example.py
```

This will demonstrate various usage patterns and show the output format.
