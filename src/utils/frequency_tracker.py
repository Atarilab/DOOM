"""
Frequency tracking utility for monitoring function call rates and performance metrics.

This module provides a reusable FrequencyTracker class that can be used to monitor
the frequency of any recurring event or function call throughout the codebase.
"""

import time
from typing import Optional, Callable, Any


class FrequencyTracker:
    """
    A utility class for tracking and logging the frequency of recurring events.
    
    This class can be used to monitor function call rates, loop frequencies,
    or any other recurring events in your application.
    
    Example usage:
        # Basic usage
        tracker = FrequencyTracker("my_function")
        for i in range(1000):
            tracker.tick()  # Call this on each event
        
        # With custom logging
        tracker = FrequencyTracker("data_processing", log_interval=10.0, logger=my_logger)
        while running:
            process_data()
            tracker.tick()
    """
    
    def __init__(
        self, 
        name: str, 
        log_interval: float = 5.0, 
        logger: Optional[Any] = None,
        log_level: str = "info",
        track_execution_time: bool = False,
        max_execution_samples: int = 100
    ):
        """
        Initialize the frequency tracker.
        
        Args:
            name: Name identifier for this tracker (used in log messages)
            log_interval: Time interval in seconds between frequency logs
            logger: Logger object to use for output (if None, uses print)
            log_level: Log level to use ("info", "debug", "warning", "error")
            track_execution_time: Whether to track execution times of function calls
            max_execution_samples: Maximum number of execution time samples to keep
        """
        self.name = name
        self.log_interval = log_interval
        self.logger = logger
        self.log_level = log_level
        self.track_execution_time = track_execution_time
        self.max_execution_samples = max_execution_samples
        
        # Tracking variables
        self._call_count = 0
        self._start_time = time.time()
        self._last_log_time = time.time()
        
        # Statistics
        self._total_calls = 0
        self._total_time = 0.0
        self._min_frequency = float('inf')
        self._max_frequency = 0.0
        
        # Execution time tracking
        self._execution_times = [] if track_execution_time else None
        
    def tick(self, count: int = 1) -> Optional[float]:
        """
        Record an event occurrence and optionally log frequency.
        
        Args:
            count: Number of events to record (default: 1)
            
        Returns:
            Current frequency in Hz if logging occurred, None otherwise
        """
        self._call_count += count
        self._total_calls += count
        current_time = time.time()
        
        # Check if it's time to log
        if current_time - self._last_log_time >= self.log_interval:
            elapsed_time = current_time - self._start_time
            if elapsed_time > 0:
                frequency = self._call_count / elapsed_time
                
                # Update statistics
                self._min_frequency = min(self._min_frequency, frequency)
                self._max_frequency = max(self._max_frequency, frequency)
                self._total_time += elapsed_time
                
                # Log the frequency
                self._log_frequency(frequency, self._call_count, elapsed_time)
                
                # Reset counters for next interval
                self._call_count = 0
                self._start_time = current_time
                self._last_log_time = current_time
                
                return frequency
        
        return None
    
    def tick_with_execution_time(self, execution_time: float) -> Optional[float]:
        """
        Record an event occurrence with execution time and optionally log frequency.
        
        Args:
            execution_time: Execution time in seconds
            
        Returns:
            Current frequency in Hz if logging occurred, None otherwise
        """
        # Record the execution time if tracking is enabled
        if self.track_execution_time and self._execution_times is not None:
            self._execution_times.append(execution_time)
            # Keep only the last N samples
            if len(self._execution_times) > self.max_execution_samples:
                self._execution_times.pop(0)
        
        # Use the regular tick method for frequency tracking
        return self.tick()
    
    def track_function(self, func: Callable, *args, **kwargs):
        """
        Execute a function and track its execution time.
        
        Args:
            func: Function to execute and track
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function execution
        """
        if not self.track_execution_time:
            # If not tracking execution time, just call tick and execute function
            self.tick()
            return func(*args, **kwargs)
        
        # Track execution time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        self.tick_with_execution_time(execution_time)
        
        return result
    
    def _log_frequency(self, frequency: float, call_count: int, elapsed_time: float):
        """Log the current frequency with appropriate method."""
        message = f"{self.name} frequency: {frequency:.2f} Hz (calls: {call_count}, time: {elapsed_time:.2f}s)"
        
        # Add execution time statistics if available
        if self.track_execution_time and self._execution_times and len(self._execution_times) > 0:
            exec_stats = self._get_execution_stats()
            if exec_stats:
                message += f" | execution: avg={exec_stats['avg_ms']:.2f}ms, min={exec_stats['min_ms']:.2f}ms, max={exec_stats['max_ms']:.2f}ms, latest={exec_stats['latest_ms']:.2f}ms"
        
        if self.logger is not None and hasattr(self.logger, self.log_level):
            log_method = getattr(self.logger, self.log_level)
            log_method(message)
        else:
            print(f"[{self.log_level.upper()}] {message}")
    
    def get_current_frequency(self) -> float:
        """
        Get the current frequency without logging.
        
        Returns:
            Current frequency in Hz based on calls since last reset
        """
        elapsed_time = time.time() - self._start_time
        if elapsed_time > 0:
            return self._call_count / elapsed_time
        return 0.0
    
    def _get_execution_stats(self) -> Optional[dict]:
        """Get execution time statistics."""
        if not self.track_execution_time or not self._execution_times or len(self._execution_times) == 0:
            return None
        
        times = self._execution_times
        return {
            "count": len(times),
            "avg_ms": sum(times) / len(times) * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
            "latest_ms": times[-1] * 1000
        }
    
    def get_execution_stats(self) -> Optional[dict]:
        """Get execution time statistics for external access."""
        return self._get_execution_stats()
    
    def get_statistics(self) -> dict:
        """
        Get comprehensive statistics about the tracked frequency.
        
        Returns:
            Dictionary containing frequency statistics
        """
        current_freq = self.get_current_frequency()
        avg_frequency = self._total_calls / self._total_time if self._total_time > 0 else 0.0
        
        stats = {
            "name": self.name,
            "current_frequency": current_freq,
            "average_frequency": avg_frequency,
            "min_frequency": self._min_frequency if self._min_frequency != float('inf') else 0.0,
            "max_frequency": self._max_frequency,
            "total_calls": self._total_calls,
            "total_time": self._total_time,
            "calls_in_current_interval": self._call_count,
            "time_in_current_interval": time.time() - self._start_time
        }
        
        # Add execution time statistics if available
        if self.track_execution_time:
            exec_stats = self._get_execution_stats()
            stats["execution_times"] = exec_stats
        
        return stats
    
    def reset_statistics(self):
        """Reset all statistics and counters."""
        self._call_count = 0
        self._start_time = time.time()
        self._last_log_time = time.time()
        self._total_calls = 0
        self._total_time = 0.0
        self._min_frequency = float('inf')
        self._max_frequency = 0.0
        
        # Reset execution times if tracking is enabled
        if self.track_execution_time and self._execution_times is not None:
            self._execution_times.clear()
    
    def set_log_interval(self, interval: float):
        """Update the logging interval."""
        self.log_interval = interval
    
    def set_logger(self, logger: Any, log_level: str = "info"):
        """Update the logger and log level."""
        self.logger = logger
        self.log_level = log_level


class MultiFrequencyTracker:
    """
    A utility class for tracking multiple frequency metrics simultaneously.
    
    Useful when you need to monitor several different events or functions
    with different logging intervals or requirements.
    
    Example usage:
        tracker = MultiFrequencyTracker()
        tracker.add_tracker("sensor_read", 1.0)  # Log every 1 second
        tracker.add_tracker("control_loop", 5.0)  # Log every 5 seconds
        
        while running:
            read_sensors()
            tracker.tick("sensor_read")
            
            control_loop()
            tracker.tick("control_loop")
    """
    
    def __init__(self, default_log_interval: float = 5.0, logger: Optional[Any] = None):
        """
        Initialize the multi-frequency tracker.
        
        Args:
            default_log_interval: Default logging interval for new trackers
            logger: Default logger for all trackers
        """
        self.trackers: dict[str, FrequencyTracker] = {}
        self.default_log_interval = default_log_interval
        self.default_logger = logger
    
    def add_tracker(
        self, 
        name: str, 
        log_interval: Optional[float] = None,
        logger: Optional[Any] = None,
        log_level: str = "info",
        track_execution_time: bool = False,
        max_execution_samples: int = 100
    ) -> FrequencyTracker:
        """
        Add a new frequency tracker.
        
        Args:
            name: Unique name for the tracker
            log_interval: Logging interval (uses default if None)
            logger: Logger for this tracker (uses default if None)
            log_level: Log level for this tracker
            track_execution_time: Whether to track execution times
            max_execution_samples: Maximum execution time samples to keep
            
        Returns:
            The created FrequencyTracker instance
        """
        if name in self.trackers:
            raise ValueError(f"Tracker '{name}' already exists")
        
        tracker = FrequencyTracker(
            name=name,
            log_interval=log_interval or self.default_log_interval,
            logger=logger or self.default_logger,
            log_level=log_level,
            track_execution_time=track_execution_time,
            max_execution_samples=max_execution_samples
        )
        self.trackers[name] = tracker
        return tracker
    
    def tick(self, name: str, count: int = 1) -> Optional[float]:
        """
        Record an event for a specific tracker.
        
        Args:
            name: Name of the tracker
            count: Number of events to record
            
        Returns:
            Current frequency if logging occurred, None otherwise
        """
        if name not in self.trackers:
            raise ValueError(f"Tracker '{name}' not found. Available trackers: {list(self.trackers.keys())}")
        
        return self.trackers[name].tick(count)
    
    def tick_with_execution_time(self, name: str, execution_time: float) -> Optional[float]:
        """
        Record an event with execution time for a specific tracker.
        
        Args:
            name: Name of the tracker
            execution_time: Execution time in seconds
            
        Returns:
            Current frequency if logging occurred, None otherwise
        """
        if name not in self.trackers:
            raise ValueError(f"Tracker '{name}' not found. Available trackers: {list(self.trackers.keys())}")
        
        return self.trackers[name].tick_with_execution_time(execution_time)
    
    def track_function(self, name: str, func: Callable, *args, **kwargs):
        """
        Execute a function and track its execution time for a specific tracker.
        
        Args:
            name: Name of the tracker
            func: Function to execute and track
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function execution
        """
        if name not in self.trackers:
            raise ValueError(f"Tracker '{name}' not found. Available trackers: {list(self.trackers.keys())}")
        
        return self.trackers[name].track_function(func, *args, **kwargs)
    
    def get_tracker(self, name: str) -> FrequencyTracker:
        """Get a specific tracker by name."""
        if name not in self.trackers:
            raise ValueError(f"Tracker '{name}' not found. Available trackers: {list(self.trackers.keys())}")
        return self.trackers[name]
    
    def get_all_statistics(self) -> dict:
        """Get statistics for all trackers."""
        return {name: tracker.get_statistics() for name, tracker in self.trackers.items()}
    
    def reset_all_statistics(self):
        """Reset statistics for all trackers."""
        for tracker in self.trackers.values():
            tracker.reset_statistics()
    
    def remove_tracker(self, name: str):
        """Remove a tracker."""
        if name in self.trackers:
            del self.trackers[name]
