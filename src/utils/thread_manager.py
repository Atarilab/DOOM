import threading
import time
from typing import Callable, List, Optional


class ThreadManager:
    """
    Generic thread management class for handling worker thread lifecycle.

    This class provides a clean interface for starting, stopping, and managing
    multiple worker threads with proper cleanup and error handling.
    """

    def __init__(self, logger=None, debug: bool = False):
        """
        Initialize the thread manager.

        :param logger: Optional logger for debugging
        :param debug: Whether to run threads in debug mode (non-daemon)
        """
        self.logger = logger
        self.debug = debug
        self._threads: List[threading.Thread] = []
        self._thread_functions: List[Callable] = []
        self._thread_names: List[str] = []
        self._running = False
        self._initialized = False

    def add_thread(self, name: str, target: Callable):
        """
        Add a worker thread to be managed.

        :param name: Name of the thread for logging
        :param target: Function to run in the thread
        """
        if self._running:
            raise RuntimeError("Cannot add threads while manager is running")

        self._thread_names.append(name)
        self._thread_functions.append(target)

    def start(self):
        """Start all registered threads."""
        if self._running:
            if self.logger:
                self.logger.debug("Threads already running, skipping start")
            return

        if not self._thread_functions:
            if self.logger:
                self.logger.warning("No threads registered to start")
            return

        # Set running flag BEFORE starting threads so they don't exit immediately
        self._running = True
        self._initialized = True

        # Create and start threads
        self._threads = []
        for name, target in zip(self._thread_names, self._thread_functions):
            thread = threading.Thread(target=target, name=name, daemon=not self.debug)
            if self.logger:
                self.logger.debug(f"Starting thread: {name}")
            thread.start()
            self._threads.append(thread)
            if self.logger:
                self.logger.debug(f"Thread {name} started, alive={thread.is_alive()}")

        if self.logger:
            self.logger.debug(f"Started {len(self._threads)} threads: {', '.join(self._thread_names)}")
            self.logger.debug(f"Thread status after start: {[thread.is_alive() for thread in self._threads]}")

    def stop(self, timeout: float = 1.0):
        """
        Stop all threads and wait for them to finish.

        :param timeout: Maximum time to wait for threads to finish
        """
        if not self._running:
            return

        self._running = False

        # Wait for threads to finish
        for thread, name in zip(self._threads, self._thread_names):
            if thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    if self.logger:
                        self.logger.warning(f"Thread {name} did not finish within {timeout}s timeout")

        self._threads.clear()

        if self.logger:
            self.logger.debug("All threads stopped")

    def is_running(self) -> bool:
        """Check if any threads are currently running."""
        running = self._running and any(thread.is_alive() for thread in self._threads)
        if self.logger:
            self.logger.debug(
                f"ThreadManager.is_running(): _running={self._running}, threads_alive={[thread.is_alive() for thread in self._threads]}, result={running}"
            )
        return running

    def should_continue(self) -> bool:
        """Check if threads should continue running."""
        return self._running

    def get_thread_status(self) -> dict:
        """
        Get status of all threads.

        :return: Dictionary with thread names and their status
        """
        status = {}
        for thread, name in zip(self._threads, self._thread_names):
            status[name] = {"alive": thread.is_alive(), "daemon": thread.daemon, "ident": thread.ident}
        return status

    def wait_for_completion(self, timeout: Optional[float] = None):
        """
        Wait for all threads to complete.

        :param timeout: Maximum time to wait (None for infinite)
        """
        if not self._running:
            return

        start_time = time.time()
        while self.is_running():
            if timeout is not None and (time.time() - start_time) > timeout:
                if self.logger:
                    self.logger.warning(f"Timeout waiting for threads to complete after {timeout}s")
                break
            time.sleep(0.01)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.stop()
        except:
            pass
