import rclpy
import logging
import time
from typing import Dict, Any, Callable, Optional
import threading
from abc import ABC, abstractmethod

class StateSubscriber(ABC):
    """Abstract base class for state subscribers.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger
    
    @abstractmethod
    def start_subscription(self):
        """Start the state subscription."""
        pass
    
    @abstractmethod
    def stop_subscription(self):
        """Stop the state subscription."""
        pass
    
    @abstractmethod
    def get_latest_state(self) -> Dict[str, Any]:
        """Retrieve the latest state."""
        pass
    
    @abstractmethod
    def spin_once(self):
        """Spin the subscriber node once."""
        pass


class DDSStateSubscriber(StateSubscriber):
    """DDS-based state subscriber."""
    
    def __init__(self, 
                 topic: str, 
                 msg_type, 
                 handler_func: Optional[Callable] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize DDS state subscriber.
        
        :param topic: DDS topic to subscribe to
        :param msg_type: Message type for the topic
        :param handler_func: Optional function to process received messages
        """
        from unitree_sdk2py.core.channel import ChannelSubscriber
        
        super().__init__(logger)
        
        self.topic = topic
        self.msg_type = msg_type
        self.handler_func = handler_func
        
        self.subscriber = ChannelSubscriber(topic, msg_type)
        self._latest_state = {}
        self._lock = threading.Lock()
    
    def _message_callback(self, msg):
        """Internal callback to store latest state with optional reordering."""
        with self._lock:
            # Extract state from message
            extracted_state = self._extract_state_from_message(msg)
            
            # Call handler if exists
            if self.handler_func:
                extracted_state = self.handler_func(extracted_state, self.logger)
            
            # Save the final state
            self._latest_state = extracted_state

                    
    def _extract_state_from_message(self, msg):
        """
        Extract state from DDS message.
        Override this method to customize state extraction.
        """
        return {
            field: getattr(msg, field) 
            for field in dir(msg) if not field.startswith('_')
        }
    
    def start_subscription(self):
        """Start DDS subscription."""
        self.subscriber.Init(self._message_callback, 10)
    
    def stop_subscription(self):
        """Stop DDS subscription."""
        # Implementation depends on Unitree SDK's unsubscribe mechanism
        pass
    
    def get_latest_state(self) -> Dict[str, Any]:
        """Retrieve the latest state."""
        with self._lock:
            return self._latest_state.copy()
        
    def spin_once(self):
        """Spin the subscriber node once. This is not required for Unitree Cyclone DDS messages."""
        pass
    

class ROS2StateSubscriber(StateSubscriber):
    """ROS2-based state subscriber with optional state reordering."""
    
    def __init__(self, 
                 topic: str, 
                 node_name: str,
                 msg_type,
                 handler_func: Optional[Callable] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize ROS2 state subscriber.
        
        :param topic: ROS2 topic to subscribe to
        :param node_name: Name of the ROS2 node
        :param msg_type: Message type for the topic
        :param handler_func: Optional function to process received messages

        """
        super().__init__(logger)
        
        self.topic = topic
        self.msg_type = msg_type
        self.handler_func = handler_func
        
        self.node = rclpy.create_node(node_name)
        self._latest_state = {}
        self._lock = threading.Lock()
        self.subscription = None
    
    def _message_callback(self, msg):
        """Internal callback to store latest state """
        with self._lock:
            # Extract state from message
            extracted_state = self._extract_state_from_message(msg)
            
            # Call handler if exists
            if self.handler_func:
                extracted_state = self.handler_func(extracted_state, self.logger)
            
            # Save the final state
            self._latest_state = extracted_state
    
    def _extract_state_from_message(self, msg):
        """
        Extract state from ROS2 message.
        Override this method to customize state extraction.
        """
        # Convert message to dictionary
        return {
            field: getattr(msg, field) 
            for field in msg.get_fields_and_field_types().keys()
        }
    
    def start_subscription(self):
        """Start ROS2 subscription."""
        self.subscription = self.node.create_subscription(
            self.msg_type, 
            self.topic, 
            self._message_callback, 
            10  # QoS depth
        )
    
    def stop_subscription(self):
        """Stop ROS2 subscription."""
        if self.subscription:
            self.node.destroy_subscription(self.subscription)
        self.node.destroy_node()
    
    def get_latest_state(self) -> Dict[str, Any]:
        """Retrieve the latest state."""
        with self._lock:
            return self._latest_state.copy()
        
    def spin_once(self):
        """Spin the subscriber node once."""
        rclpy.spin_once(self.node)
    

class StateManager:
    """
    Centralized state management for robot controllers.
    Manages multiple state subscribers and provides unified state access.
    
    Note that the state here is the states coming from the subscribed topics which are further processed
    by the ObservationManager to compute the observations for each mode/controller.
    """
    
    def __init__(self, logger=None):
        """
        Initialize State Manager.
        
        :param logger: Optional logger for tracking state updates
        """
        self._subscribers: Dict[str, StateSubscriber] = {}
        self._combined_state = {}
        self.logger = logger
    
    def add_subscriber(self, 
                       name: str, 
                       subscriber: StateSubscriber,
                       start_immediately: bool = True):
        """
        Add a state subscriber to the manager.
        
        :param name: Unique name for the subscriber
        :param subscriber: StateSubscriber instance
        :param start_immediately: Whether to start subscription immediately
        """
        if name in self._subscribers:
            raise ValueError(f"Subscriber with name {name} already exists")
        
        self._subscribers[name] = subscriber
        
        if start_immediately:
            subscriber.start_subscription()
        
        if self.logger:
            self.logger.info(f"Added state subscriber: {name}")
    
    def get_state(self, subscriber_name: str) -> Dict[str, Any]:
        """
        Retrieve state from a specific subscriber.
        
        :param subscriber_name: Name of the subscriber
        :return: Latest state dictionary
        """
        if subscriber_name not in self._subscribers:
            raise KeyError(f"No subscriber found with name {subscriber_name}")

        return self._subscribers[subscriber_name].get_latest_state()
    
    def get_combined_state(self) -> Dict[str, Any]:
        """
        Retrieve combined state from all subscribers.
        
        :return: Merged state dictionary
        """
        for name, subscriber in self._subscribers.items():
            for state_name, state_value in subscriber.get_latest_state().items():        
                self._combined_state[state_name] = state_value
        
        return self._combined_state
    
    def spin_subscribers(self):
        """
        Spin all registered subscribers once.
        """
        for name, subscriber in self._subscribers.items():
            try:
                subscriber.spin_once()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error spinning subscriber {name}: {e}")
    
    def destroy_subscribers(self):
        """
        Destroy all registered subscribers.
        """
        for name, subscriber in self._subscribers.items():
            try:
                subscriber.stop_subscription()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error destroying subscriber {name}: {e}")
        
        # Clear the subscribers dictionary
        self._subscribers.clear()
        
  