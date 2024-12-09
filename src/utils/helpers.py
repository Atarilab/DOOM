from __future__ import annotations
from typing import List

import numpy as np
import torch


class ObservationHistoryStorage:
    def __init__(self, num_envs: int, num_obs: int, max_length: int, device: torch.device = "cpu"):
        """
        Initialize a FIFO queue for state history, starting with zeros at initialization.

        Args:
            num_envs (int): Number of environments.
            num_obs (int): Number of observations per environment.
            max_length (int): Maximum length of the state history for each environment.
            device (torch.device): Device to store the buffer (e.g., "cuda" or "cpu").
        """
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.max_length = max_length
        self.device = device

        # Initialize the buffer with zeros of shape (num_envs, num_obs * max_length)
        self.buffer = torch.zeros((num_envs, num_obs * max_length), device=device)

    def add(self, observation: torch.Tensor):
        """
        Add a new observation to the buffer. Perform FIFO replacement.

        Args:
            observation (torch.Tensor): The new observation to add.
                                         Should have shape `(num_envs, num_obs)`.
        """
        if observation.shape != (self.num_envs, self.num_obs):
            raise ValueError(
                f"Observation shape must be ({self.num_envs}, {self.num_obs}). Current shape: ({observation.shape})"
            )

        # Shift the buffer to make space for the new observation
        self.buffer[:, : -self.num_obs] = self.buffer[:, self.num_obs:].clone()

        # Add the new observation at the end
        self.buffer[:, -self.num_obs:] = observation

    def get(self) -> torch.Tensor:
        """
        Get the current state history.

        Returns:
            torch.Tensor: A tensor of shape `(num_envs, num_obs * max_length)`.
        """
        return self.buffer.detach().clone()

    def reset(self, done: torch.Tensor):
        """Reset the buffer for environments that are done.

        Args:
            done (torch.Tensor): mask of dones.
        """

        done_indices = torch.nonzero(done == 1)
        self.buffer[done_indices] = 0.0


def reorder_robot_states(states: np.ndarray, origin_order: List[str], target_order: List[str]) -> np.ndarray:
    """
    Reorder robot states based on origin and target leg orders.

    Args:
        states (np.ndarray): Input states to be reordered
        origin_order (List[str]): Original leg order
        target_order (List[str]): Desired leg order

    Returns:
        np.ndarray: Reordered states
    """
    # Convert input to NumPy array if it's not already
    states = np.asarray(states)

    # Validate input lengths (4 for feet states and 12 for joint states)
    if states.size not in {4, 12}:
        raise ValueError(f"Expected 4 or 12 states, got {states.size}")

    if len(origin_order) != 4 or len(target_order) != 4:
        raise ValueError("Both origin and target orders must be lists of 4 legs")

    num_legs = 4
    # Determine number of entries per leg
    entries_per_leg = states.size // num_legs

    # Create reordering indices using NumPy
    reorder_indices = np.concatenate(
        [
            np.arange(
                origin_order.index(leg) * entries_per_leg,
                origin_order.index(leg) * entries_per_leg + entries_per_leg,
            )
            for leg in target_order
        ]
    )

    # Reorder states using advanced NumPy indexing
    return states[reorder_indices]
